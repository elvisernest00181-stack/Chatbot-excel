"""
Assistant Excel : chatbot gratuit qui répond aux questions sur un classeur
OneDrive / SharePoint. Hébergement : Streamlit Community Cloud. IA : Groq.

Fonctionnement : le classeur est chargé ici avec pandas. Pour chaque question,
l'IA écrit d'abord une petite requête (filtres, regroupements, totaux), le
programme la calcule de façon exacte sur le fichier, puis l'IA rédige la réponse
à partir de ce résultat. Les chiffres sont donc justes, et seules quelques lignes
sont envoyées à l'IA, ce qui préserve le quota gratuit de Groq.
"""
import difflib
import io
import json
import operator
import re
import unicodedata
from numbers import Number

import pandas as pd
import requests
import streamlit as st
from groq import Groq


def secret(key: str, default: str = "") -> str:
    try:
        value = st.secrets.get(key, default)
    except Exception:
        return default
    return str(value) if value not in (None, "") else default


# ---------- Réglages (à renseigner dans Settings > Secrets sur Streamlit) ----------
EXCEL_URL = secret("EXCEL_URL")
GROQ_API_KEY = secret("GROQ_API_KEY")
APP_PASSWORD = secret("APP_PASSWORD")            # optionnel : protège l'accès
APP_TITLE = secret("APP_TITLE", "Assistant Excel")
PLANNER_MODEL = secret("PLANNER_MODEL", "openai/gpt-oss-20b")
ANSWER_MODEL = secret("ANSWER_MODEL", "openai/gpt-oss-120b")
REFRESH_SECONDS = 300   # le classeur est relu au plus toutes les 5 minutes
MAX_RESULT_ROWS = 40    # nombre maximum de lignes de résultat envoyées à l'IA

st.set_page_config(page_title=APP_TITLE, page_icon="📊")

PLANNER_PROMPT = """Tu traduis une question sur un classeur Excel en requête JSON, exécutée ensuite avec pandas.
Réponds UNIQUEMENT avec un objet JSON valide, sans aucun texte autour.

Format :
{"sheet": "nom exact d'une feuille",
 "filters": [{"column": "colonne", "op": "opérateur", "value": "valeur"}],
 "group_by": ["colonne"],
 "metrics": [{"func": "count|sum|mean|min|max|nunique", "column": "colonne"}],
 "columns": ["colonnes à afficher"],
 "sort": {"by": "colonne, ou func d'une métrique", "ascending": false},
 "limit": 20}

Opérateurs : ==, !=, in, not_in (liste), contains, not_contains, >, >=, <, <=, between (liste [min, max]), is_empty, not_empty.

Règles :
- Utilise uniquement les feuilles et colonnes du schéma, avec leur orthographe exacte.
- Dates au format AAAA-MM-JJ. Une période = between. Regrouper par année ou par mois : {"column": "colonne date", "period": "year"} ou "month".
- « Combien » = metrics [{"func": "count"}] sans colonne. « Total » d'un montant = sum.
- Pour lister des lignes : metrics et group_by vides, et 3 à 8 colonnes utiles dans columns.
- Pour un texte libre (nom, ville, libellé), préfère contains à ==.
- Si la question prolonge un échange précédent, reprends ses filtres.
- Si la question ne concerne pas le classeur, renvoie {"sheet": null}.

Exemples (noms de colonnes fictifs) :
Q : Combien de sinistres en 2025 ?
{"sheet": "Sinistres", "filters": [{"column": "Date", "op": "between", "value": ["2025-01-01", "2025-12-31"]}], "metrics": [{"func": "count"}]}
Q : Montant total par agence
{"sheet": "Sinistres", "group_by": ["Agence"], "metrics": [{"func": "sum", "column": "Montant"}], "sort": {"by": "sum", "ascending": false}}"""

ANSWER_PROMPT = """Tu es l'assistant d'un classeur Excel. Réponds en français, clairement et brièvement,
uniquement à partir du résultat fourni : il a été calculé de façon exacte sur le fichier.
N'invente aucune donnée. Si le résultat est vide, dis qu'aucune ligne ne correspond aux critères.
Écris les nombres avec une espace entre les milliers (1 250 000).
Pour une liste, présente les lignes principales dans un tableau court."""

OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}
FUNC_LABELS = {"count": "nombre", "sum": "somme", "mean": "moyenne",
               "min": "minimum", "max": "maximum", "nunique": "nombre distinct"}
PERIODS = {"year": ("%Y", "année"), "month": ("%Y-%m", "mois"), "day": ("%Y-%m-%d", "jour")}
SPACES = "[\\s\u00a0\u202f]"  # espaces, y compris insécables
STOPWORDS = set(
    "les des une pour par sur dans avec sans quel quelle quels quelles combien est sont ont "
    "qui que quoi dont total liste donne moi montre affiche tous toutes tout ces cette ses "
    "leur leurs plus moins entre mon mes nos vos aux avez avoir".split()
)


# ---------- Outils de nettoyage ----------
def norm(text) -> str:
    """Minuscules, sans accents ni espaces superflus, pour comparer des textes."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip().lower()


def to_float(value) -> float:
    if isinstance(value, bool):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = re.sub(SPACES, "", str(value)).replace(",", ".")
    return float(pd.to_numeric(text, errors="coerce"))


def numeric_view(series: pd.Series):
    """Version numérique d'une colonne (accepte « 1 250 000 » ou « 12,5 »), sinon None."""
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return None
    if pd.api.types.is_numeric_dtype(series):
        return series
    present = series.notna() & (series.astype(str).str.strip() != "")
    if not present.any():
        return None
    cleaned = (series.astype(str).str.replace(SPACES, "", regex=True)
               .str.replace(",", ".", regex=False))
    converted = pd.to_numeric(cleaned, errors="coerce").where(present)
    return converted if converted[present].notna().mean() >= 0.8 else None


def to_date(value):
    text = str(value).strip()
    dayfirst = bool(re.match(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}", text))
    return pd.to_datetime(text, errors="coerce", dayfirst=dayfirst)


def fmt_number(value) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def cell(value) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.strftime("%d/%m/%Y")
    if isinstance(value, float):
        return fmt_number(value)
    return str(value)[:80]


def clean_sheet(raw: pd.DataFrame) -> pd.DataFrame:
    """Repère la ligne d'en-tête (même sous un titre) et nettoie le tableau."""
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    if raw.empty:
        return raw
    counts = raw.head(15).notna().sum(axis=1)
    header_pos = int((counts >= max(2, counts.max() * 0.7)).to_numpy().argmax())
    names, seen = [], {}
    for i, value in enumerate(raw.iloc[header_pos]):
        name = str(value).strip() if pd.notna(value) and str(value).strip() else f"Colonne {i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 0
        names.append(name)
    df = raw.iloc[header_pos + 1:].copy()
    df.columns = names
    df = df.dropna(how="all").reset_index(drop=True).infer_objects()
    for col in df.columns:  # montants saisis tantôt en nombre, tantôt en texte « 1 250 000 »
        s = df[col]
        if s.dtype == object and s.map(lambda v: isinstance(v, Number) and not isinstance(v, bool) and v == v).any():
            numbers = numeric_view(s)
            if numbers is not None:
                df[col] = numbers
    return df


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner="Lecture du classeur…")
def load_workbook(url: str) -> dict:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    if not response.content.startswith(b"PK"):
        raise ValueError(
            "le lien EXCEL_URL ne renvoie pas un fichier .xlsx. Vérifiez qu'il se termine par "
            "download=1 et qu'il s'ouvre sans connexion dans une fenêtre de navigation privée."
        )
    raw = pd.read_excel(io.BytesIO(response.content), sheet_name=None, header=None)
    sheets = {name: clean_sheet(df) for name, df in raw.items()}
    sheets = {name: df for name, df in sheets.items() if not df.empty}
    if not sheets:
        raise ValueError("le classeur ne contient aucun tableau lisible.")
    return sheets


def describe(sheets: dict) -> str:
    """Schéma compact du classeur, envoyé à l'IA à la place des données."""
    lines = []
    for name, df in sheets.items():
        lines.append(f'Feuille "{name}" ({len(df)} lignes) :')
        for col in df.columns:
            s = df[col].dropna()
            numbers = numeric_view(s) if not s.empty else None
            if s.empty:
                info = "vide"
            elif pd.api.types.is_datetime64_any_dtype(s):
                info = f"date, du {s.min():%Y-%m-%d} au {s.max():%Y-%m-%d}"
            elif numbers is not None:
                info = f"nombre, de {fmt_number(numbers.min())} à {fmt_number(numbers.max())}"
            else:
                values = s.astype(str).str.strip().unique()
                shown = values[:8] if len(values) <= 15 else values[:3]
                info = f"texte, {len(values)} valeurs distinctes, ex. : " + ", ".join(v[:40] for v in shown)
            lines.append(f"- {col} : {info}")
    return "\n".join(lines)


# ---------- Exécution exacte de la requête ----------
def find_column(df: pd.DataFrame, name):
    if name in (None, "", "*"):
        return None
    by_norm = {norm(c): c for c in df.columns}
    key = norm(name)
    if key in by_norm:
        return by_norm[key]
    close = difflib.get_close_matches(key, list(by_norm), n=1, cutoff=0.6)
    return by_norm[close[0]] if close else None


def resolve_sheet(sheets: dict, name) -> str:
    by_norm = {norm(n): n for n in sheets}
    key = norm(name or "")
    if key in by_norm:
        return by_norm[key]
    close = difflib.get_close_matches(key, list(by_norm), n=1, cutoff=0.5)
    return by_norm[close[0]] if close else next(iter(sheets))


def filter_mask(df: pd.DataFrame, flt: dict) -> pd.Series:
    col = find_column(df, flt.get("column"))
    if col is None:
        raise ValueError(f"colonne introuvable : {flt.get('column')}")
    op = str(flt.get("op", "==")).strip().lower()
    value = flt.get("value")
    values = value if isinstance(value, list) else [value]
    s = df[col]
    empty = s.isna() | (s.astype(str).str.strip() == "")
    if op == "is_empty" or (op in ("==", "=") and value is None):
        return empty
    if op == "not_empty" or (op == "!=" and value is None):
        return ~empty
    if op in ("contains", "not_contains"):
        text = s.fillna("").astype(str).map(norm)
        mask = pd.Series(False, index=df.index)
        for v in values:
            mask |= text.str.contains(norm(v), regex=False)
        return ~mask if op == "not_contains" else mask
    is_date = pd.api.types.is_datetime64_any_dtype(s)
    numbers = None if is_date else numeric_view(s)
    if op in ("==", "=", "!=", "in", "not_in"):
        if is_date:
            targets = [d.normalize() for d in map(to_date, values) if pd.notna(d)]
            mask = s.dt.normalize().isin(targets)
        elif numbers is not None and all(pd.notna(to_float(v)) for v in values):
            mask = numbers.isin([to_float(v) for v in values])
        else:
            mask = s.fillna("").astype(str).map(norm).isin([norm(v) for v in values])
        return ~mask if op in ("!=", "not_in") else mask
    if op in OPS or op == "between":
        if is_date:
            left, bounds = s, [to_date(v) for v in values]
        elif numbers is not None:
            left, bounds = numbers, [to_float(v) for v in values]
        else:
            raise ValueError(f"la colonne {col} n'est ni une date ni un nombre")
        if (op == "between" and len(bounds) < 2) or any(pd.isna(b) for b in bounds):
            raise ValueError(f"valeur invalide pour {col} : {value}")
        if op == "between":
            low, high = bounds[0], bounds[1]
            if is_date and high == high.normalize():
                return (left >= low) & (left < high + pd.Timedelta(days=1))
            return (left >= low) & (left <= high)
        bound = bounds[0]
        if is_date and op == "<=" and bound == bound.normalize():
            return left < bound + pd.Timedelta(days=1)
        return OPS[op](left, bound)
    raise ValueError(f"opérateur inconnu : {op}")


def group_key(data: pd.DataFrame, df: pd.DataFrame, item):
    if isinstance(item, dict):
        col, period = find_column(df, item.get("column")), str(item.get("period") or "").lower()
    else:
        col, period = find_column(df, item), ""
    if col is None:
        raise ValueError(f"colonne introuvable : {item}")
    s = data[col]
    if pd.api.types.is_datetime64_any_dtype(s):
        fmt, label = PERIODS.get(period, PERIODS["day"])
        return f"{col} ({label})", s.dt.strftime(fmt).fillna("(vide)"), True
    if pd.api.types.is_float_dtype(s) and (s.dropna() % 1 == 0).all():
        s = s.astype("Int64")
    return col, s.astype(object).where(s.notna(), "(vide)").astype(str).str.strip(), False


def metric_column(data: pd.DataFrame, df: pd.DataFrame, metric: dict):
    func = str(metric.get("func", "count")).strip().lower()
    if func not in FUNC_LABELS:
        raise ValueError(f"calcul inconnu : {func}")
    raw = metric.get("column")
    if func == "count" and raw in (None, "", "*"):
        return "nombre de lignes", pd.Series(1, index=data.index), "sum", func
    col = find_column(df, raw)
    if col is None:
        raise ValueError(f"colonne introuvable : {raw}")
    values = df[col]
    if func in ("sum", "mean") or (func in ("min", "max") and not pd.api.types.is_datetime64_any_dtype(values)):
        numbers = numeric_view(values)
        if numbers is None and func in ("sum", "mean"):
            raise ValueError(f"la colonne {col} ne contient pas de nombres")
        values = numbers if numbers is not None else values
    return f"{FUNC_LABELS[func]} {col}", values.loc[data.index], func, func


def run_plan(sheets: dict, plan: dict):
    name = resolve_sheet(sheets, plan.get("sheet"))
    df = sheets[name]
    mask = pd.Series(True, index=df.index)
    for flt in plan.get("filters") or []:
        mask &= filter_mask(df, flt)
    data = df[mask]

    try:
        limit = int(plan.get("limit") or 20)
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, MAX_RESULT_ROWS))
    groups, metrics = plan.get("group_by") or [], plan.get("metrics") or []
    groups = [groups] if isinstance(groups, (str, dict)) else groups
    metrics = [metrics] if isinstance(metrics, dict) else metrics
    sort = plan.get("sort") if isinstance(plan.get("sort"), dict) else {}
    sort_by = str(sort.get("by") or "").strip()
    ascending = bool(sort.get("ascending", False))

    if groups or metrics:
        frame, keys, aggs, funcs, chrono = pd.DataFrame(index=data.index), [], {}, {}, False
        for item in groups:
            label, values, is_period = group_key(data, df, item)
            frame[label] = values
            keys.append(label)
            chrono |= is_period
        for metric in metrics or [{"func": "count"}]:
            label, values, agg, func = metric_column(data, df, metric)
            frame[label] = values
            aggs[label], funcs[label] = agg, func
        if keys:
            result = frame.groupby(keys).agg(aggs).reset_index()
        else:
            result = frame[list(aggs)].agg(aggs).to_frame().T
        target = None
        if sort_by.lower() in FUNC_LABELS:
            target = next((label for label, f in funcs.items() if f == sort_by.lower()), None)
        elif sort_by:
            target = find_column(result, sort_by)
        if target is None and keys:
            target, ascending = (keys[0], True) if chrono else (next(iter(aggs)), False)
        if target is not None:
            try:
                result = result.sort_values(target, ascending=ascending)
            except TypeError:
                pass
    else:
        wanted = plan.get("columns") or []
        wanted = [wanted] if isinstance(wanted, str) else wanted
        cols = list(dict.fromkeys(c for c in (find_column(df, w) for w in wanted) if c))
        result = data[cols or list(df.columns[:8])]
        target = find_column(df, sort_by) if sort_by else None
        if target is not None:
            key = numeric_view(df[target])
            key = (key if key is not None else df[target]).loc[data.index]
            try:
                result = result.loc[key.sort_values(ascending=ascending, na_position="last").index]
            except TypeError:
                pass
    return name, len(data), result.head(limit).reset_index(drop=True)


def keyword_rows(sheets: dict, question: str):
    """Plan B si la requête échoue : lignes contenant les mots de la question."""
    words = [w for w in re.findall(r"\w+", norm(question)) if len(w) >= 3 and w not in STOPWORDS]
    best_name, best_rows = next(iter(sheets)), pd.DataFrame()
    for name, df in sheets.items():
        text = df.astype(str).apply(lambda col: col.map(norm)).agg(" ".join, axis=1)
        hits = pd.Series(0, index=df.index)
        for w in words:
            hits += text.str.contains(w, regex=False).astype(int)
        found = hits[hits > 0].sort_values(ascending=False)
        if len(found) > len(best_rows):
            best_name, best_rows = name, df.loc[found.index]
    return best_name, best_rows.head(15)


def for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Tableau affichable dans Streamlit, même si une colonne mélange les types."""
    return df.apply(lambda col: col.map(cell) if col.dtype == object else col)


def table_text(result: pd.DataFrame) -> str:
    if result is None or result.empty:
        return "(aucune ligne)"
    return result.apply(lambda col: col.map(cell)).to_csv(index=False, sep=";")


# ---------- Appels à l'IA ----------
@st.cache_resource
def groq_client():
    return Groq(api_key=GROQ_API_KEY)


def ask(model: str, system: str, user: str, temperature: float = 0.1) -> str:
    extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else None
    completion = groq_client().chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        extra_body=extra,
    )
    text = completion.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def extract_json(text: str) -> dict:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("la requête générée n'est pas du JSON valide")


def help_message(sheets: dict) -> str:
    first_name, first = next(iter(sheets.items()))
    columns = ", ".join(map(str, first.columns[:6]))
    return (
        f"Je réponds aux questions sur le classeur (feuilles : {', '.join(sheets)}). "
        f"Colonnes de « {first_name} » : {columns}…\n\n"
        "Exemples : « Combien de lignes en 2025 ? », « Total par mois », "
        "« Les 10 plus gros montants »."
    )


def answer(question: str, history: list):
    sheets = load_workbook(EXCEL_URL)
    recent = "\n".join(f"{m['role']} : {m['content'][:300]}" for m in history[-4:])
    plan_text = ask(
        PLANNER_MODEL,
        PLANNER_PROMPT + "\n\nSchéma du classeur :\n" + describe(sheets),
        (f"Échanges précédents :\n{recent}\n\n" if recent else "") + f"Question : {question}",
        temperature=0,
    )
    try:
        plan = extract_json(plan_text)
        if plan.get("sheet") is None:
            return help_message(sheets), None
        name, total, result = run_plan(sheets, plan)
        context = (f"Feuille : {name}\nRequête exécutée : {json.dumps(plan, ensure_ascii=False)}\n"
                   f"Nombre de lignes correspondant aux filtres : {total}")
    except Exception as exc:
        name, result = keyword_rows(sheets, question)
        context = (f"Feuille : {name}\nRecherche approximative par mots-clés ({exc}). "
                   "Précise que la réponse peut être incomplète.")
    reply = ask(ANSWER_MODEL, ANSWER_PROMPT,
                f"Question : {question}\n{context}\nRésultat :\n{table_text(result)}",
                temperature=0.2)
    return reply, result


def error_message(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    if status == 429:
        return ("Le quota gratuit de l'IA est atteint pour le moment. Réessayez dans une minute, "
                "ou demain si la limite du jour est dépassée.")
    if status == 413:
        return "Cette demande dépasse la taille autorisée par le palier gratuit. Posez une question plus précise."
    if status in (401, 403) and "groq" in type(exc).__module__:
        return "La clé GROQ_API_KEY est refusée. Vérifiez-la dans les secrets de l'application."
    if isinstance(exc, requests.RequestException):
        return f"Le classeur n'a pas pu être téléchargé ({exc}). Vérifiez le lien EXCEL_URL."
    return f"Une erreur est survenue : {exc}"


# ---------- Interface ----------
if not EXCEL_URL or not GROQ_API_KEY:
    st.error("Configuration incomplète : ajoutez EXCEL_URL et GROQ_API_KEY dans Settings > Secrets.")
    st.stop()

if APP_PASSWORD and not st.session_state.get("unlocked"):
    st.title(APP_TITLE)
    password = st.text_input("Mot de passe", type="password")
    if password == APP_PASSWORD:
        st.session_state.unlocked = True
        st.rerun()
    elif password:
        st.error("Mot de passe incorrect.")
    st.stop()

st.title(APP_TITLE)
st.caption("Posez vos questions sur le classeur partagé. Les chiffres sont calculés directement sur le fichier.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].replace("$", "\\$"))
        if msg.get("data") is not None:
            with st.expander("Voir les données utilisées"):
                st.dataframe(for_display(msg["data"]), hide_index=True)

if question := st.chat_input("Votre question…"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question.replace("$", "\\$"))
    with st.chat_message("assistant"):
        with st.spinner("Analyse du classeur…"):
            try:
                reply, data = answer(question, history)
            except Exception as exc:
                reply, data = error_message(exc), None
        data = data if data is not None and not data.empty else None
        st.markdown(reply.replace("$", "\\$"))
        if data is not None:
            with st.expander("Voir les données utilisées"):
                st.dataframe(for_display(data), hide_index=True)
    st.session_state.messages.append({"role": "assistant", "content": reply, "data": data})
