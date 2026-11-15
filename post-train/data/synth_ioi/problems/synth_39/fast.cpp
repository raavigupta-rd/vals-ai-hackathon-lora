#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> a(n + 1);
    int mx = 0;
    for (int i = 1; i <= n; i++) { cin >> a[i]; mx = max(mx, a[i]); }

    int B = max(1, (int)(n / max(1.0, sqrt((double)q))));
    struct Qry { int l, r, id, blk; };
    vector<Qry> qs(q);
    for (int i = 0; i < q; i++) {
        cin >> qs[i].l >> qs[i].r;
        qs[i].id = i;
        qs[i].blk = qs[i].l / B;
    }
    sort(qs.begin(), qs.end(), [](const Qry& x, const Qry& y) {
        if (x.blk != y.blk) return x.blk < y.blk;
        return (x.blk & 1) ? (x.r > y.r) : (x.r < y.r);
    });

    vector<int> cnt(mx + 1, 0);
    long long cur = 0;
    auto add = [&](int idx) {
        int x = a[idx];
        cur += 1LL * (2 * cnt[x] + 1) * x;
        cnt[x]++;
    };
    auto rem = [&](int idx) {
        int x = a[idx];
        cnt[x]--;
        cur -= 1LL * (2 * cnt[x] + 1) * x;
    };

    vector<long long> ans(q);
    int L = 1, R = 0;
    for (const auto& qr : qs) {
        while (R < qr.r) add(++R);
        while (L > qr.l) add(--L);
        while (R > qr.r) rem(R--);
        while (L < qr.l) rem(L++);
        ans[qr.id] = cur;
    }
    for (int i = 0; i < q; i++) cout << ans[i] << '\n';
    return 0;
}
