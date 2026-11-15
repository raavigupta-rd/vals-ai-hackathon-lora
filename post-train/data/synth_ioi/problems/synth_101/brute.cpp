#include <bits/stdc++.h>
using namespace std;

static const long long INF = (1LL << 60);

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k;
    cin >> n >> k;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];

    vector<long long> dp_prev(n + 1, INF), dp_cur(n + 1, INF);
    vector<int> cnt(n + 1, 0);
    dp_prev[0] = 0;
    for (int step = 1; step <= k; step++) {
        fill(dp_cur.begin(), dp_cur.end(), INF);
        for (int i = step; i <= n; i++) {
            long long pairs = 0;
            for (int j = i; j >= step; j--) {
                pairs += cnt[a[j]]++;
                if (dp_prev[j - 1] + pairs < dp_cur[i]) {
                    dp_cur[i] = dp_prev[j - 1] + pairs;
                }
            }
            for (int j = step; j <= i; j++) cnt[a[j]] = 0;
        }
        dp_prev.swap(dp_cur);
    }
    cout << dp_prev[n] << '\n';
    return 0;
}
