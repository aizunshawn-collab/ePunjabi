// Web replacement for '@react-native-async-storage/async-storage', backed
// by localStorage (synchronous under the hood, but wrapped in Promises to
// match the real library's async API).
export default {
  async getItem(key: string): Promise<string | null> {
    return window.localStorage.getItem(key);
  },
  async setItem(key: string, value: string): Promise<void> {
    window.localStorage.setItem(key, value);
  },
  async removeItem(key: string): Promise<void> {
    window.localStorage.removeItem(key);
  },
};
