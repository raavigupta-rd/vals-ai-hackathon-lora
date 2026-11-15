#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> h(n + 1);
    for (int i = 1; i <= n; i++) cin >> h[i];
    vector<long long> pref(n + 2), suf(n + 2);
    while (q--) {
        int l, r;
        cin >> l >> r;
        long long mx = 0;
        for (int i = l; i <= r; i++) {
            mx = max(mx, h[i]);
            pref[i] = mx;
        }
        mx = 0;
        long long ans = 0;
        for (int i = r; i >= l; i--) {
            mx = max(mx, h[i]);
            ans += min(pref[i], mx) - h[i];
        }
        cout << ans << '\n';
    }
    return 0;
}
