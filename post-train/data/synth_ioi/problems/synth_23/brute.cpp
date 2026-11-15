#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    while (q--) {
        int l, r;
        cin >> l >> r;
        int b[30] = {};
        for (int i = l; i <= r; i++) {
            int x = a[i];
            for (int k = 29; k >= 0; k--) if ((x >> k) & 1) {
                if (!b[k]) { b[k] = x; break; }
                x ^= b[k];
            }
        }
        int ans = 0;
        for (int k = 29; k >= 0; k--) if ((ans ^ b[k]) > ans) ans ^= b[k];
        cout << ans << '\n';
    }
    return 0;
}
