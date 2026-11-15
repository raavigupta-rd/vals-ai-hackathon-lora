#include <bits/stdc++.h>
using namespace std;

static const int FULL = (1 << 30) - 1;

struct Node {
    int all_and, all_or, mx;
    long long sum;
    int lz_and, lz_or;
    int len;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];

    vector<Node> tr(4 * n + 5);

    auto apply_uniform = [&](int u, int A, int O) {
        int new_and = (tr[u].all_and & A) | O;
        long long delta = (long long)new_and - (long long)tr[u].all_and;
        tr[u].all_and = new_and;
        tr[u].all_or = (tr[u].all_or & A) | O;
        tr[u].mx = (int)(tr[u].mx + delta);
        tr[u].sum += delta * tr[u].len;
        tr[u].lz_and &= A;
        tr[u].lz_or = (tr[u].lz_or & A) | O;
    };

    auto push = [&](int u) {
        if (tr[u].lz_and != FULL || tr[u].lz_or != 0) {
            apply_uniform(u << 1, tr[u].lz_and, tr[u].lz_or);
            apply_uniform(u << 1 | 1, tr[u].lz_and, tr[u].lz_or);
            tr[u].lz_and = FULL;
            tr[u].lz_or = 0;
        }
    };

    auto pull = [&](int u) {
        const auto& L = tr[u << 1];
        const auto& R = tr[u << 1 | 1];
        tr[u].all_and = L.all_and & R.all_and;
        tr[u].all_or = L.all_or | R.all_or;
        tr[u].mx = max(L.mx, R.mx);
        tr[u].sum = L.sum + R.sum;
    };

    auto build = [&](auto& self, int u, int l, int r) -> void {
        tr[u].lz_and = FULL;
        tr[u].lz_or = 0;
        tr[u].len = r - l + 1;
        if (l == r) {
            tr[u].all_and = tr[u].all_or = tr[u].mx = a[l];
            tr[u].sum = a[l];
            return;
        }
        int m = (l + r) >> 1;
        self(self, u << 1, l, m);
        self(self, u << 1 | 1, m + 1, r);
        pull(u);
    };

    auto upd_and = [&](auto& self, int u, int l, int r, int ql, int qr, int x) -> void {
        int clr = FULL ^ x;
        if ((tr[u].all_or & clr) == 0) return;
        if (ql <= l && r <= qr) {
            int diff = tr[u].all_or ^ tr[u].all_and;
            if ((diff & clr) == 0) {
                apply_uniform(u, x, 0);
                return;
            }
        }
        push(u);
        int m = (l + r) >> 1;
        if (ql <= m) self(self, u << 1, l, m, ql, qr, x);
        if (qr > m) self(self, u << 1 | 1, m + 1, r, ql, qr, x);
        pull(u);
    };

    auto upd_or = [&](auto& self, int u, int l, int r, int ql, int qr, int x) -> void {
        if (((FULL ^ tr[u].all_and) & x) == 0) return;
        if (ql <= l && r <= qr) {
            int diff = tr[u].all_or ^ tr[u].all_and;
            if ((diff & x) == 0) {
                apply_uniform(u, FULL, x);
                return;
            }
        }
        push(u);
        int m = (l + r) >> 1;
        if (ql <= m) self(self, u << 1, l, m, ql, qr, x);
        if (qr > m) self(self, u << 1 | 1, m + 1, r, ql, qr, x);
        pull(u);
    };

    auto query = [&](auto& self, int u, int l, int r, int ql, int qr) -> pair<int, long long> {
        if (ql <= l && r <= qr) return {tr[u].mx, tr[u].sum};
        push(u);
        int m = (l + r) >> 1;
        pair<int, long long> res{-1, 0};
        if (ql <= m) {
            auto lv = self(self, u << 1, l, m, ql, qr);
            res.first = max(res.first, lv.first);
            res.second += lv.second;
        }
        if (qr > m) {
            auto rv = self(self, u << 1 | 1, m + 1, r, ql, qr);
            res.first = max(res.first, rv.first);
            res.second += rv.second;
        }
        return res;
    };

    build(build, 1, 1, n);

    while (q--) {
        int tp, l, r;
        cin >> tp >> l >> r;
        if (tp == 1) {
            int x;
            cin >> x;
            upd_and(upd_and, 1, 1, n, l, r, x);
        } else if (tp == 2) {
            int x;
            cin >> x;
            upd_or(upd_or, 1, 1, n, l, r, x);
        } else {
            auto [mx, sm] = query(query, 1, 1, n, l, r);
            cout << mx << ' ' << sm << '\n';
        }
    }
    return 0;
}
