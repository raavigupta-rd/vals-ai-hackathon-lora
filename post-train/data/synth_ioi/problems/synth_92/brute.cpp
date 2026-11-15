#include <bits/stdc++.h>
using namespace std;

static constexpr int MOD = 998244353;

// 40-pt subtask solver: Manacher O(N) + O(N^2) DP
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string s;
    cin >> s;
    int n = (int)s.size();

    vector<int> d1(n), d2(n);
    for (int i = 0, l = 0, r = -1; i < n; i++) {
        int k = (i > r) ? 1 : min(d1[l + r - i], r - i + 1);
        while (0 <= i - k && i + k < n && s[i - k] == s[i + k]) k++;
        d1[i] = k--;
        if (i + k > r) { l = i - k; r = i + k; }
    }
    for (int i = 0, l = 0, r = -1; i < n; i++) {
        int k = (i > r) ? 0 : min(d2[l + r - i + 1], r - i + 1);
        while (0 <= i - k - 1 && i + k < n && s[i - k - 1] == s[i + k]) k++;
        d2[i] = k--;
        if (i + k > r) { l = i - k - 1; r = i + k; }
    }
    auto is_pal = [&](int l, int r) -> bool {
        int len = r - l + 1;
        int mid = (l + r) >> 1;
        if (len & 1) return d1[mid] >= (len + 1) / 2;
        return d2[mid + 1] >= len / 2;
    };

    vector<int> dp(n + 1, 0);
    dp[0] = 1;
    for (int i = 1; i <= n; i++) {
        long long tot = 0;
        for (int j = 0; j <= i - 2; j++) {
            if (dp[j] && is_pal(j, i - 1)) tot += dp[j];
        }
        dp[i] = (int)(tot % MOD);
    }
    cout << dp[n] << '\n';
    return 0;
}
