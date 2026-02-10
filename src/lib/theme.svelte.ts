export type ThemeMode = "dark" | "light";
export type LayoutMode = "centered" | "stretched";
export type FontFamily = "jetbrains" | "cascadia" | "fira" | "system" | "inter";
export type FontSize = 10 | 11 | 12 | 13 | 14;

export const FONT_FAMILIES: { id: FontFamily; name: string; css: string }[] = [
  { id: "jetbrains", name: "JETBRAINS MONO", css: '"JetBrains Mono", monospace' },
  { id: "cascadia", name: "CASCADIA CODE", css: '"Cascadia Code", monospace' },
  { id: "fira", name: "FIRA CODE", css: '"Fira Code", monospace' },
  { id: "system", name: "SYSTEM MONO", css: '"SF Mono", "Consolas", "DejaVu Sans Mono", "Liberation Mono", monospace' },
  { id: "inter", name: "INTER (SANS)", css: '"Inter", "Segoe UI", "Helvetica Neue", sans-serif' },
];

class ThemeStore {
  mode = $state<ThemeMode>("dark");
  layout = $state<LayoutMode>("stretched");
  fontFamily = $state<FontFamily>("jetbrains");
  fontSize = $state<FontSize>(12);

  toggleMode() {
    this.mode = this.mode === "dark" ? "light" : "dark";
  }

  setMode(m: ThemeMode) {
    this.mode = m;
  }

  setLayout(l: LayoutMode) {
    this.layout = l;
  }

  setFontFamily(f: FontFamily) {
    this.fontFamily = f;
  }

  setFontSize(s: FontSize) {
    this.fontSize = s;
  }

  get fontFamilyCss(): string {
    return FONT_FAMILIES.find(f => f.id === this.fontFamily)?.css
      ?? FONT_FAMILIES[0].css;
  }
}

export const theme = new ThemeStore();
