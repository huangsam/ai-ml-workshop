/**
 * Client-side Python code syntax highlighting
 */

export const escapeHtml = (text: string): string => {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#x27;");
};

/**
 * Custom client-side Python syntax highlighter (zero-dependency, lightweight, performant, bug-free)
 */
export const highlightPython = (rawCode: string): string => {
  interface Token {
    type: "comment" | "string" | "keyword" | "builtin" | "number" | "text";
    val: string;
  }

  const tokens: Token[] = [];
  let pos = 0;
  const len = rawCode.length;

  // Matches from the start of the remaining string
  const commentRegex = /^#[^\n]*/;
  const stringRegex = /^("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/;
  const keywordRegex =
    /^(def|class|import|from|return|if|else|elif|for|while|in|is|not|and|or|try|except|finally|with|as|lambda|pass|global|nonlocal|del|yield|async|await|match|case|None|True|False)\b/;
  const builtinRegex =
    /^(main|fit|predict|train|evaluate|save_plot|update_stage|update_metrics|is_cancelled|backward|step|zero_grad|forward|optim|DataLoader)\b/;
  const numberRegex = /^(\d+(?:\.\d+)?)\b/;
  const wordRegex = /^[a-zA-Z_]\w*/;
  const whitespaceRegex = /^\s+/;

  while (pos < len) {
    const remaining = rawCode.slice(pos);

    // 1. Whitespace
    let match = whitespaceRegex.exec(remaining);
    if (match) {
      tokens.push({ type: "text", val: match[0] });
      pos += match[0].length;
      continue;
    }

    // 2. Comments
    match = commentRegex.exec(remaining);
    if (match) {
      tokens.push({ type: "comment", val: match[0] });
      pos += match[0].length;
      continue;
    }

    // 3. Strings
    match = stringRegex.exec(remaining);
    if (match) {
      tokens.push({ type: "string", val: match[0] });
      pos += match[0].length;
      continue;
    }

    // 4. Keywords
    match = keywordRegex.exec(remaining);
    if (match) {
      tokens.push({ type: "keyword", val: match[0] });
      pos += match[0].length;
      continue;
    }

    // 5. Builtins / ML functions
    match = builtinRegex.exec(remaining);
    if (match) {
      tokens.push({ type: "builtin", val: match[0] });
      pos += match[0].length;
      continue;
    }

    // 6. Numbers
    match = numberRegex.exec(remaining);
    if (match) {
      tokens.push({ type: "number", val: match[0] });
      pos += match[0].length;
      continue;
    }

    // 7. Standard identifiers
    match = wordRegex.exec(remaining);
    if (match) {
      tokens.push({ type: "text", val: match[0] });
      pos += match[0].length;
      continue;
    }

    // 8. Single characters
    tokens.push({ type: "text", val: rawCode[pos] });
    pos++;
  }

  // Convert tokens to styled HTML
  return tokens
    .map((t) => {
      const escapedVal = escapeHtml(t.val);
      if (t.type === "comment") {
        return `<span class="text-gray-500 italic">${escapedVal}</span>`;
      }
      if (t.type === "string") {
        return `<span class="text-emerald-400 font-medium">${escapedVal}</span>`;
      }
      if (t.type === "keyword") {
        return `<span class="text-indigo-400 font-semibold">${escapedVal}</span>`;
      }
      if (t.type === "builtin") {
        return `<span class="text-sky-400 font-medium">${escapedVal}</span>`;
      }
      if (t.type === "number") {
        return `<span class="text-pink-400">${escapedVal}</span>`;
      }
      return escapedVal;
    })
    .join("");
};
