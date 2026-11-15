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
    const int INF = 1e9;
    vector<int> dp(S + 1, INF);
    dp[0] = 0;
    int cur = 0;
    for (int w : a) {
        cur += w;
        for (int s = cur; s >= w; s--) {
            int cand = dp[s - w] + 1;
            if (cand < dp[s]) dp[s] = cand;
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
