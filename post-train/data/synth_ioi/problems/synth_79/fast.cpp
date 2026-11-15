#include <bits/stdc++.h>
using namespace std;

static constexpr long long MOD = 998244353LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];

    vector<long long> sum(4 * n + 5, 0), mul(4 * n + 5, 1), add(4 * n + 5, 0);

    auto apply = [&](int u, int len, long long b, long long c) {
        sum[u] = (sum[u] * b + c * len) % MOD;
        mul[u] = (mul[u] * b) % MOD;
        add[u] = (add[u] * b + c) % MOD;
    };
    auto push = [&](int u, int l, int r) {
        if (mul[u] != 1 || add[u] != 0) {
            int mid = (l + r) >> 1;
            apply(u << 1, mid - l + 1, mul[u], add[u]);
            apply(u << 1 | 1, r - mid, mul[u], add[u]);
            mul[u] = 1;
            add[u] = 0;
        }
    };

    function<void(int, int, int)> build = [&](int u, int l, int r) {
        if (l == r) { sum[u] = a[l] % MOD; return; }
        int mid = (l + r) >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
        sum[u] = (sum[u << 1] + sum[u << 1 | 1]) % MOD;
    };
    function<void(int, int, int, int, int, long long, long long)> upd =
        [&](int u, int l, int r, int ql, int qr, long long b, long long c) {
            if (ql <= l && r <= qr) { apply(u, r - l + 1, b, c); return; }
            push(u, l, r);
            int mid = (l + r) >> 1;
            if (ql <= mid) upd(u << 1, l, mid, ql, qr, b, c);
            if (qr > mid) upd(u << 1 | 1, mid + 1, r, ql, qr, b, c);
            sum[u] = (sum[u << 1] + sum[u << 1 | 1]) % MOD;
        };
    function<long long(int, int, int, int, int)> qry =
        [&](int u, int l, int r, int ql, int qr) -> long long {
            if (ql <= l && r <= qr) return sum[u];
            push(u, l, r);
            int mid = (l + r) >> 1;
            long long res = 0;
            if (ql <= mid) res += qry(u << 1, l, mid, ql, qr);
            if (qr > mid) res += qry(u << 1 | 1, mid + 1, r, ql, qr);
            return res % MOD;
        };

    build(1, 1, n);
    while (q--) {
        int t; cin >> t;
        if (t == 1) {
            int l, r; long long b, c;
            cin >> l >> r >> b >> c;
            upd(1, 1, n, l, r, b, c);
        } else {
            int l, r;
            cin >> l >> r;
            cout << qry(1, 1, n, l, r) << '\n';
        }
    }
    return 0;
}
