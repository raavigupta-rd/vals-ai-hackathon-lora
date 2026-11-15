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
        int tp, l, r;
        cin >> tp >> l >> r;
        if (tp == 1) {
            int x;
            cin >> x;
            for (int i = l; i <= r; i++) a[i] &= x;
        } else if (tp == 2) {
            int x;
            cin >> x;
            for (int i = l; i <= r; i++) a[i] |= x;
        } else {
            int mx = 0;
            long long sm = 0;
            for (int i = l; i <= r; i++) {
                mx = max(mx, a[i]);
                sm += a[i];
            }
            cout << mx << ' ' << sm << '\n';
        }
    }
    return 0;
}
