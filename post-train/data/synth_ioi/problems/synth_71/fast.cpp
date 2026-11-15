#include <bits/stdc++.h>
using namespace std;

struct Event {
    int x, y1, y2, delta;
    bool operator<(const Event& o) const { return x < o.x; }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<Event> evs;
    evs.reserve(2 * n);
    vector<int> ys;
    ys.reserve(2 * n);
    for (int i = 0; i < n; i++) {
        int x1, y1, x2, y2;
        cin >> x1 >> y1 >> x2 >> y2;
        evs.push_back({x1, y1, y2, +1});
        evs.push_back({x2, y1, y2, -1});
        ys.push_back(y1);
        ys.push_back(y2);
    }
    sort(ys.begin(), ys.end());
    ys.erase(unique(ys.begin(), ys.end()), ys.end());
    sort(evs.begin(), evs.end());

    int m = (int)ys.size() - 1; // elementary intervals 0 .. m-1
    if (m <= 0) {
        cout << 0 << '\n';
        return 0;
    }

    vector<int> cnt(4 * m + 5, 0);
    vector<long long> cov(4 * m + 5, 0);

    function<void(int, int, int, int, int, int)> upd = [&](int u, int l, int r, int ql, int qr, int d) {
        if (ql <= l && r <= qr) {
            cnt[u] += d;
        } else {
            int mid = (l + r) >> 1;
            if (ql <= mid) upd(u << 1, l, mid, ql, qr, d);
            if (qr > mid) upd(u << 1 | 1, mid + 1, r, ql, qr, d);
        }
        if (cnt[u] > 0) cov[u] = (long long)ys[r + 1] - ys[l];
        else if (l == r) cov[u] = 0;
        else cov[u] = cov[u << 1] + cov[u << 1 | 1];
    };

    long long ans = 0;
    for (size_t i = 0; i < evs.size(); ) {
        int cur_x = evs[i].x;
        if (i > 0) {
            long long dx = (long long)cur_x - evs[i - 1].x;
            ans += dx * cov[1];
        }
        while (i < evs.size() && evs[i].x == cur_x) {
            int l = (int)(lower_bound(ys.begin(), ys.end(), evs[i].y1) - ys.begin());
            int r = (int)(lower_bound(ys.begin(), ys.end(), evs[i].y2) - ys.begin()) - 1;
            if (l <= r) upd(1, 0, m - 1, l, r, evs[i].delta);
            i++;
        }
    }
    cout << ans << '\n';
    return 0;
}
