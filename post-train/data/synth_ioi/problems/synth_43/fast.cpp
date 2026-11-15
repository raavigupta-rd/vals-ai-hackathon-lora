#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<pair<int, int>> items(n);
    for (int i = 0; i < n; i++) { cin >> items[i].first; items[i].second = i + 1; }
    struct Qry { int l, r, x, id; };
    vector<Qry> qs(q);
    for (int i = 0; i < q; i++) { cin >> qs[i].l >> qs[i].r >> qs[i].x; qs[i].id = i; }

    sort(items.begin(), items.end());
    sort(qs.begin(), qs.end(), [](const Qry& a, const Qry& b) { return a.x < b.x; });

    vector<long long> bit(n + 1, 0), ans(q);
    auto add = [&](int idx, long long v) {
        for (; idx <= n; idx += idx & -idx) bit[idx] += v;
    };
    auto sum = [&](int idx) {
        long long s = 0;
        for (; idx > 0; idx -= idx & -idx) s += bit[idx];
        return s;
    };

    int p = 0;
    for (const auto& qr : qs) {
        while (p < n && items[p].first <= qr.x) {
            add(items[p].second, items[p].first);
            p++;
        }
        ans[qr.id] = sum(qr.r) - sum(qr.l - 1);
    }
    for (int i = 0; i < q; i++) cout << ans[i] << '\n';
    return 0;
}
