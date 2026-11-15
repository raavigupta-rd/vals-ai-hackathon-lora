#include <bits/stdc++.h>
using namespace std;

struct Pt {
    long long x, y;
    int half() const { return (y > 0 || (y == 0 && x > 0)) ? 0 : 1; }
};

static inline __int128 cross(const Pt& a, const Pt& b) {
    return (__int128)a.x * b.y - (__int128)a.y * b.x;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<Pt> p(n);
    for (int i = 0; i < n; i++) cin >> p[i].x >> p[i].y;
    sort(p.begin(), p.end(), [](const Pt& a, const Pt& b) {
        int ha = a.half(), hb = b.half();
        if (ha != hb) return ha < hb;
        return cross(a, b) > 0;
    });
    p.resize(2 * n);
    for (int i = 0; i < n; i++) p[n + i] = p[i];

    long long total = (__int128)n * (n - 1) * (n - 2) / 6;
    long long bad = 0;
    int r = 0;
    for (int i = 0; i < n; i++) {
        if (r < i) r = i;
        while (r + 1 < i + n && cross(p[i], p[r + 1]) > 0) r++;
        long long m = r - i;
        bad += m * (m - 1) / 2;
    }
    cout << (total - bad) << '\n';
    return 0;
}
