#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    struct Seg { int l, r; long long w; };
    vector<Seg> a(m);
    for (int i = 0; i < m; i++) cin >> a[i].l >> a[i].r >> a[i].w;
    sort(a.begin(), a.end(), [](const Seg& x, const Seg& y) { return x.r < y.r; });

    const long long INF = (1LL << 61);
    vector<long long> dp(n + 1, INF);
    dp[0] = 0;
    for (const auto& s : a) {
        long long best = INF;
        for (int x = s.l - 1; x <= s.r; x++) best = min(best, dp[x]);
        if (best < INF) dp[s.r] = min(dp[s.r], best + s.w);
    }
    cout << (dp[n] >= INF ? -1LL : dp[n]) << '\n';
    return 0;
}
