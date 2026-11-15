#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    const long long NEG = -(1LL << 60);
    vector<long long> dp(n + 1, NEG);
    dp[0] = 0;
    int best = 0;
    for (int i = 1; i <= n; i++) {
        long long x;
        cin >> x;
        for (int k = best + 1; k >= 1; k--) {
            if (dp[k - 1] != NEG && dp[k - 1] + x >= 0) {
                dp[k] = max(dp[k], dp[k - 1] + x);
                if (k > best) best = k;
            }
        }
        cout << best << (i == n ? '\n' : ' ');
    }
    return 0;
}
