#include <bits/stdc++.h>
using namespace std;

struct Info {
    int len, pref, suff, best;
};

static inline Info merge_info(const Info& a, const Info& b) {
    if (a.len == 0) return b;
    if (b.len == 0) return a;
    Info res;
    res.len = a.len + b.len;
    res.pref = (a.pref == a.len) ? (a.len + b.pref) : a.pref;
    res.suff = (b.suff == b.len) ? (b.len + a.suff) : b.suff;
    res.best = max({a.best, b.best, a.suff + b.pref});
    return res;
}

struct Node {
    int lc, rc;
    int pref, suff, best;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;
    vector<pair<int, int>> ord(n);
    for (int i = 1; i <= n; i++) {
        int h;
        cin >> h;
        ord[i - 1] = {h, i};
    }
    sort(ord.begin(), ord.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });

    vector<Node> tr;
    tr.reserve(n * 20 + 4 * n + 10);
    tr.push_back({0, 0, 0, 0, 0});

    auto build = [&](auto& self, int l, int r) -> int {
        int u = (int)tr.size();
        tr.push_back({0, 0, 0, 0, 0});
        if (l == r) return u;
        int m = (l + r) >> 1;
        int lc = self(self, l, m);
        int rc = self(self, m + 1, r);
        tr[u].lc = lc;
        tr[u].rc = rc;
        return u;
    };

    auto update = [&](auto& self, int prev, int l, int r, int pos) -> int {
        int u = (int)tr.size();
        tr.push_back(tr[prev]);
        if (l == r) {
            tr[u].pref = tr[u].suff = tr[u].best = 1;
            return u;
        }
        int m = (l + r) >> 1;
        if (pos <= m) {
            int lc = self(self, tr[prev].lc, l, m, pos);
            tr[u].lc = lc;
        } else {
            int rc = self(self, tr[prev].rc, m + 1, r, pos);
            tr[u].rc = rc;
        }
        int lc = tr[u].lc, rc = tr[u].rc;
        int llen = m - l + 1, rlen = r - m;
        tr[u].pref = (tr[lc].pref == llen) ? (llen + tr[rc].pref) : tr[lc].pref;
        tr[u].suff = (tr[rc].suff == rlen) ? (rlen + tr[lc].suff) : tr[rc].suff;
        tr[u].best = max({tr[lc].best, tr[rc].best, tr[lc].suff + tr[rc].pref});
        return u;
    };

    auto query = [&](auto& self, int u, int l, int r, int ql, int qr) -> Info {
        if (ql <= l && r <= qr) {
            return {r - l + 1, tr[u].pref, tr[u].suff, tr[u].best};
        }
        int m = (l + r) >> 1;
        if (qr <= m) return self(self, tr[u].lc, l, m, ql, qr);
        if (ql > m) return self(self, tr[u].rc, m + 1, r, ql, qr);
        return merge_info(self(self, tr[u].lc, l, m, ql, qr),
                          self(self, tr[u].rc, m + 1, r, ql, qr));
    };

    vector<int> roots(n + 1, 0);
    roots[0] = build(build, 1, n);
    for (int i = 1; i <= n; i++) {
        roots[i] = update(update, roots[i - 1], 1, n, ord[i - 1].second);
    }

    while (q--) {
        int l, r, w;
        cin >> l >> r >> w;
        int lo = 1, hi = n, ans_idx = n;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (query(query, roots[mid], 1, n, l, r).best >= w) {
                ans_idx = mid;
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
        cout << ord[ans_idx - 1].first << '\n';
    }
    return 0;
}
