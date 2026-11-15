#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> a(n + 1), b(n + 1);
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
        b[i] = a[i];
    }
    while (q--) {
        int tp, l, r;
        cin >> tp >> l >> r;
        if (tp == 1) {
            long long x;
            cin >> x;
            for (int i = l; i <= r; i++) {
                a[i] += x;
                if (a[i] > b[i]) b[i] = a[i];
            }
        } else if (tp == 2) {
            long long v;
            cin >> v;
            for (int i = l; i <= r; i++) {
                if (v > a[i]) a[i] = v;
                if (a[i] > b[i]) b[i] = a[i];
            }
        } else if (tp == 3) {
            long long v;
            cin >> v;
            for (int i = l; i <= r; i++) {
                a[i] = v;
                if (a[i] > b[i]) b[i] = a[i];
            }
        } else {
            long long ma = -(1LL << 60), mb = -(1LL << 60);
            for (int i = l; i <= r; i++) {
                ma = max(ma, a[i]);
                mb = max(mb, b[i]);
            }
            cout << ma << ' ' << mb << '\n';
        }
    }
    return 0;
}
