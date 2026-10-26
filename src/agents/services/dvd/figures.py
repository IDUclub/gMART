"""Figure evidence and permanent links to the authenticated DVD image proxy."""

import re
from urllib.parse import quote, urlencode

FIGURE_RULE = (
    "Фрагмент «Рисунок» описывает изображение. Его описание составлено моделью "
    "по картинке, а не является нормативным текстом. Числа из описания — результат "
    "машинного чтения изображения; при их использовании прямо укажи это и цитируй "
    "рисунок. Не представляй описание как дословную норму документа. "
)
FIGURE_CRITIC_RULE = (
    FIGURE_RULE + "\n"
    "Numbers stated in a figure's machine-generated description are supported by that\n"
    "cited fragment. They describe the picture and must not be presented as normative text.\n"
)
MAX_FIGURES = 6
_NUMBER = re.compile(r"^рисунок\s+(?:№\s*)?([А-ЯA-Z]?\.?\d+(?:\.\d+)*)", re.I)


def is_figure(hit):
    return any(hit.get(key) == "figure" for key in ("kind", "type", "element_type"))


def caption(hit):
    return str((hit.get("figure") or {}).get("caption") or "").strip()


def figure_address(hit):
    title = " ".join(caption(hit).split())
    if match := _NUMBER.match(title):
        return f"рисунок {match[1]}"
    return f"рисунок «{title}»" if title else "рисунок"


def figure_text(hit):
    title = caption(hit)
    description = str((hit.get("figure") or {}).get("description") or "").strip()
    # Older search envelopes may carry the description only in the hit text.
    if not description:
        description = str(hit.get("source_text") or hit.get("text") or "").strip()
        if title and description.startswith(title):
            description = description[len(title) :].strip()
    heading = f"Рисунок: {title}" if title else "Рисунок без подписи"
    return heading + (
        "\nОписание изображения (составлено моделью по картинке): " + description
        if description
        else ""
    )


def cited_figures(answer, hits):
    from .dvd_context import DvdContextBuilder

    sources = dict(enumerate(DvdContextBuilder.ordered_hits(hits), 1))
    seen, result = set(), []
    generated = "\n".join(
        line for line in answer.splitlines() if not line.lstrip().startswith(">")
    )
    for label in re.findall(r"\[(\d+)\]", generated):
        hit = sources.get(int(label), {})
        node_id = hit.get("node_id") or hit.get("id")
        if is_figure(hit) and node_id and node_id not in seen:
            seen.add(node_id)
            result.append(hit)
            if len(result) == MAX_FIGURES:
                break
    return result


def figure_file(hit, *, project_id=None, scenario_id=None):
    node_id = str(hit.get("node_id") or hit["id"])
    url = "/documents/figures/" + quote(node_id, safe="")
    params = {}
    if hit.get("user_id") or hit.get("owner") == "user":
        project = hit.get("project_id") or project_id
        if project is not None:
            params["project_id"] = str(project)
        elif scenario_id is not None:
            params["scenario_id"] = str(scenario_id)
    if params:
        url += "?" + urlencode(params)
    filename = f"{hit.get('name') or 'document'}-{figure_address(hit)}"
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)
    filename = filename[:150].strip(" .") + ".png"
    return {
        "name": "figure",
        "title": caption(hit) or "Рисунок",
        "role": "figure",
        "url": url,
        "download_url": url + ("&" if params else "?") + "download=1",
        "filename": filename,
        "mime_type": "image/png",
        "source_service": "idu_dvd",
    }
