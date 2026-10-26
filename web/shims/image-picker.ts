// Web replacement for 'react-native-image-picker'. There is no OS camera/
// gallery picker API in a browser - this triggers a hidden
// <input type="file" accept="image/*"> instead, using the "capture"
// attribute to hint at opening the device camera directly when the real app
// asked for the camera. Resolves to the same {didCancel, errorCode, assets}
// shape TranslationAppMobile.tsx already reads.
import {fileToBase64} from './virtual-fs';

type PickerOptions = {mediaType?: 'photo'; includeBase64?: boolean; quality?: number};
type PickerAsset = {base64?: string};
type PickerResponse = {didCancel?: boolean; errorCode?: string; assets?: PickerAsset[]};

export type Asset = PickerAsset;

function pickImage(useCameraCapture: boolean): Promise<PickerResponse> {
  return new Promise(resolve => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    if (useCameraCapture) {
      input.setAttribute('capture', 'environment');
    }
    input.style.display = 'none';

    let settled = false;
    const finish = (response: PickerResponse) => {
      if (settled) {
        return;
      }
      settled = true;
      document.body.removeChild(input);
      resolve(response);
    };

    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) {
        finish({didCancel: true});
        return;
      }
      const base64 = await fileToBase64(file);
      finish({assets: [{base64}]});
    };
    // No reliable browser "cancel" event for file inputs - if the user
    // closes the dialog without picking anything, onchange never fires and
    // the caller simply never resolves beyond the implicit didCancel state
    // the window regaining focus below approximates.
    window.addEventListener(
      'focus',
      () => setTimeout(() => finish({didCancel: true}), 300),
      {once: true},
    );

    document.body.appendChild(input);
    input.click();
  });
}

export async function launchCamera(_options: PickerOptions): Promise<PickerResponse> {
  return pickImage(true);
}

export async function launchImageLibrary(_options: PickerOptions): Promise<PickerResponse> {
  return pickImage(false);
}
