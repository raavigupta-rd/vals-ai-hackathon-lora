#include <bits/stdc++.h>
using namespace std;

struct Node {
    int lc, rc, sum;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;
    vector<long long> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];

    vector<Node> tr;
    tr.reserve(n * 40 + 10);
    tr.push_back({0, 0, 0});

    auto upd = [&](auto& self, int prev, int l, int r, int pos, int delta) -> int {
        int u = (int)tr.size();
        tr.push_back(tr[prev]);
        tr[u].sum += delta;
        if (l == r) return u;
        int m = (l + r) >> 1;
        if (pos <= m) {
            int lc = self(self, tr[prev].lc, l, m, pos, delta);
            tr[u].lc = lc;
        } else {
            int rc = self(self, tr[prev].rc, m + 1, r, pos, delta);
            tr[u].rc = rc;
        }
        return u;
    };

    auto pref_sum = [&](auto& self, int u, int l, int r, int qr) -> int {
        if (!u || l > qr) return 0;
        if (r <= qr) return tr[u].sum;
        int m = (l + r) >> 1;
        return self(self, tr[u].lc, l, m, qr) + self(self, tr[u].rc, m + 1, r, qr);
    };

    auto kth_pos = [&](auto& self, int u, int l, int r, int k) -> int {
        if (l == r) return l;
        int m = (l + r) >> 1;
        int lsum = tr[tr[u].lc].sum;
        if (k <= lsum) return self(self, tr[u].lc, l, m, k);
        return self(self, tr[u].rc, m + 1, r, k - lsum);
    };

    // Coordinate compress for O(1) last_seen lookup
    vector<long long> vals(a.begin() + 1, a.end());
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    vector<int> last_pos(vals.size(), 0);

    vector<int> roots(n + 2, 0);
    for (int i = n; i >= 1; i--) {
        int cid = (int)(lower_bound(vals.begin(), vals.end(), a[i]) - vals.begin());
        int cur = roots[i + 1];
        if (last_pos[cid] != 0) {
            cur = upd(upd, cur, 1, n, last_pos[cid], -1);
        }
        cur = upd(upd, cur, 1, n, i, +1);
        last_pos[cid] = i;
        roots[i] = cur;
    }

    long long last_ans = 0;
    while (q--) {
        long long x, y, z;
        cin >> x >> y >> z;
        int l = (int)((x + last_ans) % n) + 1;
        int r = (int)((y + last_ans) % n) + 1;
        if (l > r) swap(l, r);
        int len = r - l + 1;
        int k = (int)((z + last_ans) % len) + 1;

        int cnt = pref_sum(pref_sum, roots[l], 1, n, r);
        if (cnt < k) {
            cout << -1 << '\n';
            last_ans = 0;
        } else {
            int idx = kth_pos(kth_pos, roots[l], 1, n, k);
            cout << a[idx] << '\n';
            last_ans = a[idx];
        }
    }
    return 0;
}
