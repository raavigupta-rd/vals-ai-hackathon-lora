#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> a(n), vals;
    for (int i = 0; i < n; i++) {
        cin >> a[i];
        a[i] -= (i + 1);
        vals.push_back(a[i]);
    }
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    int m = (int)vals.size();
    vector<long long> dp(m, 0);
    for (int i = 0; i < n; i++) {
        long long best = (1LL << 62);
        for (int j = 0; j < m; j++) {
            best = min(best, dp[j]);
            dp[j] = best + llabs(a[i] - vals[j]);
        }
    }
    cout << *min_element(dp.begin(), dp.end()) << '\n';
    return 0;
}
