#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    int S = 0;
    vector<int> a(n);
    for (int i = 0; i < n; i++) {
        cin >> a[i];
        S += a[i];
    }
    vector<int> freq(S + 1, 0);
    for (int w : a) freq[w]++;

    vector<pair<int, int>> bundles;
    for (int w = 1; w <= S; w++) {
        int c = freq[w];
        for (int k = 1; c > 0; k <<= 1) {
            int take = min(k, c);
            bundles.push_back({take * w, take});
            c -= take;
        }
    }

    const int INF = 1e9;
    vector<int> dp(S + 1, INF);
    dp[0] = 0;
    int cur = 0;
    for (auto [bw, bc] : bundles) {
        cur += bw;
        for (int s = cur; s >= bw; s--) {
            dp[s] = min(dp[s], dp[s - bw] + bc);
        }
    }

    long long cnt_ok = 0, sum_f = 0;
    for (int s = 1; s <= S; s++) {
        if (dp[s] < INF) {
            cnt_ok++;
            sum_f += dp[s];
        }
    }
    cout << cnt_ok << ' ' << sum_f << '\n';
    return 0;
}
