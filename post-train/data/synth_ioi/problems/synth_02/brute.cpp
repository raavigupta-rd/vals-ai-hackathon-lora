#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    vector<int> cnt(n + 1, 0), touched;
    while (q--) {
        int t;
        cin >> t;
        if (t == 1) {
            int p, x;
            cin >> p >> x;
            a[p] = x;
        } else {
            int l, r;
            cin >> l >> r;
            int ans = 0, need = (r - l + 1) / 2;
            touched.clear();
            for (int i = l; i <= r; i++) {
                if (cnt[a[i]] == 0) touched.push_back(a[i]);
                if (++cnt[a[i]] > need) ans = a[i];
            }
            for (int c : touched) cnt[c] = 0;
            cout << ans << '\n';
        }
    }
    return 0;
}
