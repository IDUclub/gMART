import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = { children: ReactNode };
type State = { error: Error | null };

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Ошибка интерфейса gMART", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <main className="fatal-error" role="alert">
        <section>
          <span>Интерфейс остановлен</span>
          <h1>Не удалось отобразить рабочее пространство</h1>
          <p>{this.state.error.message || "Неизвестная ошибка интерфейса"}</p>
          <button onClick={() => window.location.reload()}>
            Перезагрузить страницу
          </button>
        </section>
      </main>
    );
  }
}
