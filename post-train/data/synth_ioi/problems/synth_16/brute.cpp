#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> a(n + 1), d(n + 1), f(n + 1), dp(n + 1, 0);
    for (int i = 1; i <= n; i++) cin >> a[i];
    for (int i = 1; i <= n; i++) cin >> d[i];
    for (int i = 1; i <= n; i++) cin >> f[i];
    for (int i = 2; i <= n; i++) {
        long long best = (1LL << 62);
        for (int j = 1; j < i; j++) best = min(best, dp[j] + a[j] * d[i]);
        dp[i] = best + f[i];
        cout << dp[i] << '\n';
    }
    return 0;
}
