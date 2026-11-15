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
            int p; long long x; cin >> p >> x;
            a[p] = x;
        } else {
            int l, r; cin >> l >> r;
            long long cur = a[l], best = a[l];
            for (int i = l + 1; i <= r; i++) {
                cur = max(a[i], cur + a[i]);
                best = max(best, cur);
            }
            cout << best << '\n';
        }
    }
    return 0;
}
