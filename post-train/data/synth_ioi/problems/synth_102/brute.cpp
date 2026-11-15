#include <bits/stdc++.h>
using namespace std;

static const int MOD = 998244353;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int b, q;
    cin >> b >> q;
    int N = 1 << b;
    vector<int> f(N), g(N);
    for (int i = 0; i < N; i++) {
        cin >> f[i];
        f[i] %= MOD;
    }
    for (int i = 0; i < N; i++) {
        cin >> g[i];
        g[i] %= MOD;
    }
    while (q--) {
        int s;
        cin >> s;
        long long ans = 0;
        for (int t = s; ; t = (t - 1) & s) {
            ans = (ans + 1LL * f[t] * g[s ^ t]) % MOD;
            if (t == 0) break;
        }
        cout << ans << '\n';
    }
    return 0;
}
