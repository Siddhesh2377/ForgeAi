export type ThemeMode = "dark" | "light";
export type LayoutMode = "centered" | "stretched";

class ThemeStore {
  mode = $state<ThemeMode>("dark");
  layout = $state<LayoutMode>("stretched");

  toggleMode() {
    this.mode = this.mode === "dark" ? "light" : "dark";
  }

  setMode(m: ThemeMode) {
    this.mode = m;
  }

  setLayout(l: LayoutMode) {
    this.layout = l;
  }
}

export const theme = new ThemeStore();
