#include <bits/stdc++.h>
using namespace std;

static const long long NEG_INF = -(1LL << 60);

struct Tag {
    long long a, b; // f(x) = max(x + a, b)
};

static inline long long add_clamped(long long x, long long y) {
    if (x <= NEG_INF / 2 || y <= NEG_INF / 2) return NEG_INF;
    long long s = x + y;
    if (s <= NEG_INF) return NEG_INF;
    return s;
}

static inline long long eval_tag(const Tag& t, long long x) {
    return max(add_clamped(x, t.a), t.b);
}

// g(f(x)) = max(max(x + f.a, f.b) + g.a, g.b)
static inline Tag compose_tag(const Tag& g, const Tag& f) {
    return {add_clamped(f.a, g.a), max(add_clamped(f.b, g.a), g.b)};
}

static inline Tag max_tag(const Tag& f, const Tag& g) {
    return {max(f.a, g.a), max(f.b, g.b)};
}

struct SegNode {
    long long max_a, max_b;
    Tag cur, hist;
    bool dirty;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;
    vector<long long> init_a(n + 1);
    for (int i = 1; i <= n; i++) cin >> init_a[i];

    const Tag ID{0, NEG_INF};
    vector<SegNode> tr(4 * n + 5, {NEG_INF, NEG_INF, ID, ID, false});

    auto apply_window = [&](int u, const Tag& w_cur, const Tag& w_hist) {
        long long old_a = tr[u].max_a;
        tr[u].max_a = eval_tag(w_cur, old_a);
        tr[u].max_b = max(tr[u].max_b, eval_tag(w_hist, old_a));
        if (!tr[u].dirty) {
            tr[u].cur = w_cur;
            tr[u].hist = w_hist;
            tr[u].dirty = true;
        } else {
            tr[u].hist = max_tag(tr[u].hist, compose_tag(w_hist, tr[u].cur));
            tr[u].cur = compose_tag(w_cur, tr[u].cur);
        }
    };

    auto push = [&](int u) {
        if (!tr[u].dirty) return;
        apply_window(u << 1, tr[u].cur, tr[u].hist);
        apply_window(u << 1 | 1, tr[u].cur, tr[u].hist);
        tr[u].cur = ID;
        tr[u].hist = ID;
        tr[u].dirty = false;
    };

    auto pull = [&](int u) {
        tr[u].max_a = max(tr[u << 1].max_a, tr[u << 1 | 1].max_a);
        tr[u].max_b = max(tr[u << 1].max_b, tr[u << 1 | 1].max_b);
    };

    auto build = [&](auto& self, int u, int l, int r) -> void {
        if (l == r) {
            tr[u].max_a = init_a[l];
            tr[u].max_b = init_a[l];
            return;
        }
        int m = (l + r) >> 1;
        self(self, u << 1, l, m);
        self(self, u << 1 | 1, m + 1, r);
        pull(u);
    };

    auto update = [&](auto& self, int u, int l, int r, int ql, int qr, const Tag& tg) -> void {
        if (ql <= l && r <= qr) {
            apply_window(u, tg, tg);
            return;
        }
        push(u);
        int m = (l + r) >> 1;
        if (ql <= m) self(self, u << 1, l, m, ql, qr, tg);
        if (qr > m) self(self, u << 1 | 1, m + 1, r, ql, qr, tg);
        pull(u);
    };

    auto query = [&](auto& self, int u, int l, int r, int ql, int qr) -> pair<long long, long long> {
        if (ql <= l && r <= qr) return {tr[u].max_a, tr[u].max_b};
        push(u);
        int m = (l + r) >> 1;
        pair<long long, long long> res{NEG_INF, NEG_INF};
        if (ql <= m) {
            auto lv = self(self, u << 1, l, m, ql, qr);
            res.first = max(res.first, lv.first);
            res.second = max(res.second, lv.second);
        }
        if (qr > m) {
            auto rv = self(self, u << 1 | 1, m + 1, r, ql, qr);
            res.first = max(res.first, rv.first);
            res.second = max(res.second, rv.second);
        }
        return res;
    };

    build(build, 1, 1, n);

    while (q--) {
        int tp, l, r;
        cin >> tp >> l >> r;
        if (tp == 1) {
            long long x;
            cin >> x;
            update(update, 1, 1, n, l, r, {x, NEG_INF});
        } else if (tp == 2) {
            long long v;
            cin >> v;
            update(update, 1, 1, n, l, r, {0, v});
        } else if (tp == 3) {
            long long v;
            cin >> v;
            update(update, 1, 1, n, l, r, {NEG_INF, v});
        } else {
            auto [ma, mb] = query(query, 1, 1, n, l, r);
            cout << ma << ' ' << mb << '\n';
        }
    }
    return 0;
}
