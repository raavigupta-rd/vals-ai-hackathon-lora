#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, C, q;
    cin >> n >> C >> q;
    vector<int> w(n + 1);
    vector<long long> v(n + 1);
    for (int i = 1; i <= n; i++) cin >> w[i] >> v[i];
    vector<long long> dp(C + 1);
    while (q--) {
        int k, b;
        cin >> k >> b;
        fill(dp.begin(), dp.begin() + b + 1, 0LL);
        for (int i = 1; i <= n; i++) if (i != k && w[i] <= b) {
            for (int c = b; c >= w[i]; c--) dp[c] = max(dp[c], dp[c - w[i]] + v[i]);
        }
        cout << dp[b] << '\n';
    }
    return 0;
}
