#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    while (q--) {
        int t; cin >> t;
        if (t == 1) {
            int l, r; long long m; cin >> l >> r >> m;
            for (int i = l; i <= r; i++) a[i] %= m;
        } else if (t == 2) {
            int p; long long x; cin >> p >> x;
            a[p] = x;
        } else {
            int l, r; cin >> l >> r;
            long long s = 0;
            for (int i = l; i <= r; i++) s += a[i];
            cout << s << '\n';
        }
    }
    return 0;
}
