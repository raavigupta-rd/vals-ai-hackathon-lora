#include <bits/stdc++.h>
using namespace std;

static constexpr int MOD = 1'000'000'007;

struct Val {
    int len = 0;
    int cnt = 0;
};

static inline Val merge_val(Val a, Val b) {
    if (a.len != b.len) return a.len > b.len ? a : b;
    int s = a.cnt + b.cnt;
    if (s >= MOD) s -= MOD;
    return {a.len, s};
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];
    vector<int> vals = a;
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    int m = (int)vals.size();

    vector<Val> bit(m + 1);
    auto query = [&](int idx) {
        Val res{0, 0};
        for (; idx > 0; idx -= idx & -idx) res = merge_val(res, bit[idx]);
        return res;
    };
    auto update = [&](int idx, Val v) {
        for (; idx <= m; idx += idx & -idx) bit[idx] = merge_val(bit[idx], v);
    };

    for (int x : a) {
        int rk = (int)(lower_bound(vals.begin(), vals.end(), x) - vals.begin()) + 1;
        Val best = query(rk - 1);
        Val cur = (best.len == 0) ? Val{1, 1} : Val{best.len + 1, best.cnt};
        update(rk, cur);
    }
    Val ans = query(m);
    cout << ans.len << ' ' << ans.cnt << '\n';
    return 0;
}
