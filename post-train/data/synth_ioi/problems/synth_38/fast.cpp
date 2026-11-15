#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> a(n + 1), vals;
    vals.reserve(n);
    for (int i = 1; i <= n; i++) { cin >> a[i]; vals.push_back(a[i]); }
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    int m = (int)vals.size();

    int max_nodes = (n + 5) * 20;
    vector<int> lc(max_nodes, 0), rc(max_nodes, 0), cnt(max_nodes, 0);
    vector<long long> sm(max_nodes, 0);
    int ptr = 0;

    function<int(int, int, int, int, long long)> upd =
        [&](int prev, int l, int r, int pos, long long v) -> int {
            int u = ++ptr;
            lc[u] = lc[prev];
            rc[u] = rc[prev];
            cnt[u] = cnt[prev] + 1;
            sm[u] = sm[prev] + v;
            if (l == r) return u;
            int mid = (l + r) >> 1;
            if (pos <= mid) lc[u] = upd(lc[prev], l, mid, pos, v);
            else rc[u] = upd(rc[prev], mid + 1, r, pos, v);
            return u;
        };

    vector<int> root(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        int id = (int)(lower_bound(vals.begin(), vals.end(), a[i]) - vals.begin());
        root[i] = upd(root[i - 1], 0, m - 1, id, a[i]);
    }

    while (q--) {
        int l, r, k;
        cin >> l >> r >> k;
        int uL = root[l - 1], uR = root[r];
        int lo = 0, hi = m - 1;
        long long s = 0;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            int cleft = cnt[lc[uR]] - cnt[lc[uL]];
            if (k <= cleft) {
                uL = lc[uL];
                uR = lc[uR];
                hi = mid;
            } else {
                k -= cleft;
                s += sm[lc[uR]] - sm[lc[uL]];
                uL = rc[uL];
                uR = rc[uR];
                lo = mid + 1;
            }
        }
        long long kth = vals[lo];
        s += 1LL * k * kth;
        cout << kth << ' ' << s << '\n';
    }
    return 0;
}
