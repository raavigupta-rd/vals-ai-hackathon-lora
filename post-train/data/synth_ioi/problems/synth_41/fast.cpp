#include <bits/stdc++.h>
using namespace std;

struct SegTree {
    int n;
    vector<int> mx;
    SegTree(const vector<int>& a) {
        n = (int)a.size() - 1;
        mx.assign(4 * n + 4, 0);
        build(1, 1, n, a);
    }
    void build(int u, int l, int r, const vector<int>& a) {
        if (l == r) { mx[u] = a[l]; return; }
        int m = (l + r) >> 1;
        build(2 * u, l, m, a);
        build(2 * u + 1, m + 1, r, a);
        mx[u] = max(mx[2 * u], mx[2 * u + 1]);
    }
    void set_val(int u, int l, int r, int p, int v) {
        if (l == r) { mx[u] = v; return; }
        int m = (l + r) >> 1;
        if (p <= m) set_val(2 * u, l, m, p, v);
        else set_val(2 * u + 1, m + 1, r, p, v);
        mx[u] = max(mx[2 * u], mx[2 * u + 1]);
    }
    int find_first(int u, int l, int r, int ql, int v) const {
        if (r < ql || mx[u] < v) return -1;
        if (l == r) return l;
        int m = (l + r) >> 1;
        int res = find_first(2 * u, l, m, ql, v);
        if (res != -1) return res;
        return find_first(2 * u + 1, m + 1, r, ql, v);
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    SegTree st(a);
    while (q--) {
        int t, x, y;
        cin >> t >> x >> y;
        if (t == 1) st.set_val(1, 1, n, x, y);
        else cout << st.find_first(1, 1, n, x, y) << '\n';
    }
    return 0;
}
