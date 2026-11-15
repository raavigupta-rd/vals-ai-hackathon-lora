#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> x(n), h(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> h[i];

    int S = 1;
    while (S < n) S <<= 1;
    vector<int> seg(2 * S, 0);
    auto seg_set = [&](int idx, int val) {
        int u = S + idx;
        seg[u] = val;
        for (u >>= 1; u >= 1; u >>= 1) seg[u] = max(seg[2 * u], seg[2 * u + 1]);
    };
    auto seg_q = [&](int l, int r) {
        int res = l;
        for (l += S, r += S + 1; l < r; l >>= 1, r >>= 1) {
            if (l & 1) res = max(res, seg[l++]);
            if (r & 1) res = max(res, seg[--r]);
        }
        return res;
    };

    vector<int> ans(n);
    for (int i = n - 1; i >= 0; i--) {
        int reach = (int)(upper_bound(x.begin(), x.end(), x[i] + h[i]) - x.begin()) - 1;
        int far = max(reach, seg_q(i, reach));
        seg_set(i, far);
        ans[i] = far - i + 1;
    }
    for (int i = 0; i < n; i++) cout << ans[i] << (i + 1 == n ? '\n' : ' ');
    return 0;
}
