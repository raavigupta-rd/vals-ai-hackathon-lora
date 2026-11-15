#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k;
    cin >> n >> k;
    vector<long long> p(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        long long x; cin >> x;
        p[i] = p[i - 1] + x;
    }
    const long long INF = (1LL << 62);
    vector<long long> dp(n + 1, INF), ndp(n + 1, INF);
    dp[0] = 0;
    for (int step = 1; step <= k; step++) {
        fill(ndp.begin(), ndp.end(), INF);
        for (int i = step; i <= n; i++) {
            for (int j = step - 1; j < i; j++) {
                long long d = p[i] - p[j];
                ndp[i] = min(ndp[i], dp[j] + d * d);
            }
        }
        dp.swap(ndp);
    }
    cout << dp[n] << '\n';
    return 0;
}
