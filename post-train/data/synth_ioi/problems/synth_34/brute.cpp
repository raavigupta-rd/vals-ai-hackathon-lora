#include <bits/stdc++.h>
using namespace std;

const long long MOD = 998244353LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> m(n);
    vector<long long> w(n);
    for (int i = 0; i < n; i++) { cin >> m[i] >> w[i]; w[i] %= MOD; }
    vector<long long> ans(400005, 0);
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            int s = m[i] + m[j];
            ans[s] = (ans[s] + w[i] * w[j]) % MOD;
        }
    }
    while (q--) {
        int s; cin >> s;
        cout << ans[s] << '\n';
    }
    return 0;
}
