// Web replacement for react-native-web's Alert, which is a hard no-op
// (`static alert() {}`) - there's no native dialog API in a browser, and
// window.confirm() only supports two buttons (not the 3-4 option action
// sheets TranslationAppMobile.tsx uses for "Scan Photo"/"Translate Document"
// source pickers), so this renders a minimal DOM-based modal instead,
// matching the real Alert.alert(title, message, buttons) signature.
type AlertButton = {
  text?: string;
  onPress?: () => void;
  style?: 'default' | 'cancel' | 'destructive';
};

export default class Alert {
  static alert(title?: string, message?: string, buttons?: AlertButton[]): void {
    const resolvedButtons: AlertButton[] = buttons && buttons.length > 0 ? buttons : [{text: 'OK'}];

    const overlay = document.createElement('div');
    overlay.style.cssText =
      'position:fixed;inset:0;background:rgba(0,0,0,0.5);display:flex;align-items:center;' +
      'justify-content:center;z-index:9999;';

    const box = document.createElement('div');
    box.style.cssText =
      'background:#fff;border-radius:10px;padding:20px;min-width:260px;max-width:90vw;' +
      'box-shadow:0 4px 24px rgba(0,0,0,0.3);font-family:inherit;';

    if (title) {
      const heading = document.createElement('div');
      heading.textContent = title;
      heading.style.cssText = 'font-weight:600;font-size:16px;margin-bottom:8px;color:#111;';
      box.appendChild(heading);
    }
    if (message) {
      const body = document.createElement('div');
      body.textContent = message;
      body.style.cssText = 'font-size:14px;margin-bottom:16px;color:#333;';
      box.appendChild(body);
    }

    const buttonColumn = document.createElement('div');
    buttonColumn.style.cssText = 'display:flex;flex-direction:column;gap:8px;';

    const close = () => {
      document.body.removeChild(overlay);
      document.removeEventListener('keydown', onKeyDown);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        close();
      }
    };
    document.addEventListener('keydown', onKeyDown);

    resolvedButtons.forEach(button => {
      const el = document.createElement('button');
      el.textContent = button.text ?? 'OK';
      const isCancel = button.style === 'cancel';
      const isDestructive = button.style === 'destructive';
      el.style.cssText =
        'padding:10px 14px;border-radius:6px;border:none;cursor:pointer;font-size:14px;' +
        `background:${isDestructive ? '#d32f2f' : isCancel ? '#eee' : '#1a73e8'};` +
        `color:${isCancel ? '#333' : '#fff'};`;
      el.onclick = () => {
        close();
        button.onPress?.();
      };
      buttonColumn.appendChild(el);
    });

    box.appendChild(buttonColumn);
    overlay.appendChild(box);
    overlay.onclick = event => {
      if (event.target === overlay) {
        close();
      }
    };
    document.body.appendChild(overlay);
  }
}
