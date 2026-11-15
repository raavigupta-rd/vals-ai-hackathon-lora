#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    vector<long long> c(n + 1);
    for (int i = 1; i <= n; i++) cin >> c[i];
    vector<vector<pair<int, long long>>> by_r(n + 1);
    for (int i = 0; i < m; i++) {
        int l, r; long long p;
        cin >> l >> r >> p;
        by_r[r].push_back({l, p});
    }

    vector<long long> f(n + 1, 0), gain_at_l(n + 1, 0);
    for (int r = 1; r <= n; r++) {
        for (auto [l, p] : by_r[r]) gain_at_l[l] += p;
        long long best = f[r - 1];
        long long cur_gain = 0, cur_cost = 0;
        for (int j = r; j >= 1; j--) {
            cur_gain += gain_at_l[j];
            cur_cost += c[j];
            long long prev = (j >= 2 ? f[j - 2] : 0LL);
            best = max(best, prev + cur_gain - cur_cost);
        }
        f[r] = best;
    }
    cout << f[n] << '\n';
    return 0;
}
