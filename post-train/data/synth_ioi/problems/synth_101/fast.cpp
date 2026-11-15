#include <bits/stdc++.h>
using namespace std;

static const long long INF = (1LL << 60);

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k;
    if (!(cin >> n >> k)) return 0;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];

    vector<int> cnt(n + 1, 0);
    int curL = 1, curR = 0;
    long long cur_cost = 0;

    auto get_cost = [&](int L, int R) -> long long {
        while (curL > L) {
            --curL;
            cur_cost += cnt[a[curL]]++;
        }
        while (curR < R) {
            ++curR;
            cur_cost += cnt[a[curR]]++;
        }
        while (curL < L) {
            cur_cost -= --cnt[a[curL]];
            ++curL;
        }
        while (curR > R) {
            cur_cost -= --cnt[a[curR]];
            --curR;
        }
        return cur_cost;
    };

    vector<long long> dp_prev(n + 1, INF), dp_cur(n + 1, INF);
    dp_prev[0] = 0;
    for (int i = 1; i <= n; i++) dp_prev[i] = get_cost(1, i);

    auto solve = [&](auto& self, int l, int r, int optL, int optR) -> void {
        if (l > r) return;
        int mid = (l + r) >> 1;
        long long best_val = INF;
        int best_j = optL;
        int hi = min(mid - 1, optR);
        for (int j = optL; j <= hi; j++) {
            long long cand = dp_prev[j] + get_cost(j + 1, mid);
            if (cand < best_val) {
                best_val = cand;
                best_j = j;
            }
        }
        dp_cur[mid] = best_val;
        self(self, l, mid - 1, optL, best_j);
        self(self, mid + 1, r, best_j, optR);
    };

    for (int step = 2; step <= k; step++) {
        fill(dp_cur.begin(), dp_cur.end(), INF);
        solve(solve, step, n, step - 1, n - 1);
        dp_prev.swap(dp_cur);
    }
    cout << dp_prev[n] << '\n';
    return 0;
}
