#include <bits/stdc++.h>
using namespace std;

static const long long NEG_INF = -(1LL << 62);

struct Line {
    long long a, b;
};

static inline bool bad_upper(const Line& l1, const Line& l2, const Line& l3) {
    // Upper hull of y = a*x + b with a1 < a2 < a3: l2 is redundant iff
    // (b2 - b1)/(a1 - a2) >= (b3 - b2)/(a2 - a3) <=> (b2 - b1)*(a3 - a2) <= (b3 - b2)*(a2 - a1)
    return (__int128)(l2.b - l1.b) * (l3.a - l2.a) <= (__int128)(l3.b - l2.b) * (l2.a - l1.a);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int q;
    if (!(cin >> q)) return 0;

    struct Interval {
        long long a, b;
        int l, r;
    };
    vector<Interval> items;
    vector<int> open_idx;
    vector<pair<int, long long>> queries;

    for (int t = 1; t <= q; t++) {
        int tp;
        cin >> tp;
        if (tp == 1) {
            long long a, b;
            cin >> a >> b;
            open_idx.push_back((int)items.size());
            items.push_back({a, b, t, q});
        } else if (tp == 2) {
            int id;
            cin >> id;
            items[open_idx[id - 1]].r = t - 1;
        } else {
            long long x;
            cin >> x;
            queries.push_back({t, x});
        }
    }

    if (queries.empty()) return 0;

    // Sort lines by slope ascending once so each SegTree node builds hull in O(sz)
    sort(items.begin(), items.end(), [](const Interval& u, const Interval& v) {
        if (u.a != v.a) return u.a < v.a;
        return u.b < v.b;
    });

    vector<vector<Line>> tr(4 * q + 5);
    auto add_line = [&](auto& self, int u, int l, int r, int ql, int qr, const Line& ln) -> void {
        if (ql > qr) return;
        if (ql <= l && r <= qr) {
            auto& h = tr[u];
            if (!h.empty() && h.back().a == ln.a) {
                h.back().b = ln.b; // sorted ascending in b, so ln.b >= h.back().b
            } else {
                while (h.size() >= 2 && bad_upper(h[h.size() - 2], h.back(), ln)) h.pop_back();
                h.push_back(ln);
            }
            return;
        }
        int m = (l + r) >> 1;
        if (ql <= m) self(self, u << 1, l, m, ql, qr, ln);
        if (qr > m) self(self, u << 1 | 1, m + 1, r, ql, qr, ln);
    };

    for (const auto& it : items) {
        if (it.l <= it.r) add_line(add_line, 1, 1, q, it.l, it.r, {it.a, it.b});
    }

    // Sort queries by x ascending for O(1) amortized pointer advance per node
    int m = (int)queries.size();
    vector<int> qord(m);
    iota(qord.begin(), qord.end(), 0);
    sort(qord.begin(), qord.end(), [&](int i, int j) {
        return queries[i].second < queries[j].second;
    });

    vector<int> ptr(4 * q + 5, 0);
    auto eval_node = [&](int u, long long x) -> long long {
        const auto& h = tr[u];
        if (h.empty()) return NEG_INF;
        int& p = ptr[u];
        while (p + 1 < (int)h.size() && h[p].a * x + h[p].b <= h[p + 1].a * x + h[p + 1].b) {
            p++;
        }
        return h[p].a * x + h[p].b;
    };

    vector<long long> ans(m, NEG_INF);
    for (int idx : qord) {
        int t = queries[idx].first;
        long long x = queries[idx].second;
        int u = 1, l = 1, r = q;
        long long best = NEG_INF;
        while (true) {
            best = max(best, eval_node(u, x));
            if (l == r) break;
            int mid = (l + r) >> 1;
            if (t <= mid) {
                u = u << 1;
                r = mid;
            } else {
                u = u << 1 | 1;
                l = mid + 1;
            }
        }
        ans[idx] = best;
    }

    for (int i = 0; i < m; i++) {
        if (ans[i] == NEG_INF) cout << "EMPTY\n";
        else cout << ans[i] << '\n';
    }
    return 0;
}
