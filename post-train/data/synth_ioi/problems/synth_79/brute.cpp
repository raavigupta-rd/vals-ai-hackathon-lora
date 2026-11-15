#include <bits/stdc++.h>
using namespace std;

static constexpr long long MOD = 998244353LL;

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
            int l, r; long long b, c;
            cin >> l >> r >> b >> c;
            for (int i = l; i <= r; i++) a[i] = (b * a[i] + c) % MOD;
        } else {
            int l, r;
            cin >> l >> r;
            long long s = 0;
            for (int i = l; i <= r; i++) s += a[i];
            cout << (s % MOD) << '\n';
        }
    }
    return 0;
}
