#include <bits/stdc++.h>
using namespace std;

const long long INF = (1LL << 62);
struct Line {
    long long m, b;
    long long eval(long long x) const { return m * x + b; }
};

struct LiChao {
    int XMAX;
    vector<Line> tr;
    LiChao(int xmax) : XMAX(xmax), tr(4 * xmax + 8, {0, INF}) {}
    void add_line(int u, int l, int r, Line nw) {
        int mid = (l + r) >> 1;
        bool lef = nw.eval(l) < tr[u].eval(l);
        bool md = nw.eval(mid) < tr[u].eval(mid);
        if (md) swap(tr[u], nw);
        if (l == r) return;
        if (lef != md) add_line(2 * u, l, mid, nw);
        else add_line(2 * u + 1, mid + 1, r, nw);
    }
    long long query(int u, int l, int r, int x) const {
        long long res = tr[u].eval(x);
        if (l == r) return res;
        int mid = (l + r) >> 1;
        if (x <= mid) return min(res, query(2 * u, l, mid, x));
        return min(res, query(2 * u + 1, mid + 1, r, x));
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> a(n + 1), d(n + 1), f(n + 1), dp(n + 1, 0);
    for (int i = 1; i <= n; i++) cin >> a[i];
    for (int i = 1; i <= n; i++) cin >> d[i];
    for (int i = 1; i <= n; i++) cin >> f[i];

    LiChao lc(1000000);
    lc.add_line(1, 1, 1000000, {a[1], 0});
    for (int i = 2; i <= n; i++) {
        dp[i] = f[i] + lc.query(1, 1, 1000000, (int)d[i]);
        lc.add_line(1, 1, 1000000, {a[i], dp[i]});
        cout << dp[i] << '\n';
    }
    return 0;
}
