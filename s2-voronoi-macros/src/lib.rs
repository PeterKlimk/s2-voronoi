use proc_macro::{Delimiter, Group, Ident, Literal, Punct, Spacing, TokenStream, TokenTree};

/// Generates a `clip_convex_*` function for an N-gon with a jump table over all contiguous inside runs.
///
/// Invocation:
/// - `gen_clip_convex_ngon!(clip_convex_pent, 5);`
///
/// The generated function expects the following items to be in scope at the expansion site:
/// - `PolyBuffer`, `HalfPlane`, `ClipResult`
/// - `clip_convex_small_bool::<N>(...)` as a cold fallback
#[proc_macro]
pub fn gen_clip_convex_ngon(input: TokenStream) -> TokenStream {
    impl_gen_clip_convex_ngon(input, false)
}

/// Generates a SIMD-optimized `clip_convex_*` function for an N-gon.
///
/// Uses `std::simd::f64x8` for classification. Requires `N <= 8`.
///
/// Invocation:
/// - `gen_clip_convex_simd_ngon!(clip_convex_pent_simd, 5);`
#[proc_macro]
pub fn gen_clip_convex_simd_ngon(input: TokenStream) -> TokenStream {
    impl_gen_clip_convex_ngon(input, true)
}

fn impl_gen_clip_convex_ngon(input: TokenStream, simd: bool) -> TokenStream {
    let (fn_name, n) = match parse_ident_and_usize(input) {
        Ok(v) => v,
        Err(e) => return e,
    };

    if !(3..=8).contains(&n) {
        return compile_error(&format!("N must be in 3..=8, got {n}"));
    }

    let full_mask: u16 = (1u16 << n) - 1;

    let mut cases: Vec<(u8, String)> = Vec::new();
    for start in 0..n {
        for len in 1..n {
            let mut mask: u8 = 0;
            let mut verts: Vec<usize> = Vec::with_capacity(len);
            for k in 0..len {
                let idx = (start + k) % n;
                verts.push(idx);
                mask |= 1u8 << idx;
            }

            // Entry edge is (prev -> start), where prev is outside and start is inside.
            let prev = (start + n - 1) % n;
            // Exit edge is (last -> next), where last is inside and next is outside.
            let last = (start + len - 1) % n;
            let next = (last + 1) % n;

            // Emit:
            // push_entry!(start, prev, prev);
            // push_v!(...);
            // push_exit!(last, next, last);
            let mut body = String::new();
            body.push_str(&format!("push_entry!({start}, {prev}, {prev});\n"));
            for &v in &verts {
                body.push_str(&format!("push_v!({v});\n"));
            }
            body.push_str(&format!("push_exit!({last}, {next}, {last});\n"));

            cases.push((mask, body));
        }
    }

    // Sort deterministically.
    cases.sort_by_key(|(mask, _)| *mask);
    cases.dedup_by_key(|(mask, _)| *mask);

    let mut match_arms = String::new();
    for (mask, body) in cases {
        // Skip empty/full patterns (handled by early-outs).
        if mask == 0 || mask as u16 == full_mask {
            continue;
        }
        match_arms.push_str(&format!("0b{mask:0width$b} => {{ {body} }}\n", width = n));
    }

    let classification_code = if simd {
        format!(
            r#"
    use std::simd::prelude::*;
    use std::simd::{{f64x8, num::SimdFloat}};

    // SAFETY: PolyBuffer has static size >= 8 (64 actually), so this load is always valid.
    let us_simd = f64x8::from_slice(&poly.us[0..8]);
    let vs_simd = f64x8::from_slice(&poly.vs[0..8]);

    let a_simd = f64x8::splat(hp.a);
    let b_simd = f64x8::splat(hp.b);
    let c_simd = f64x8::splat(hp.c);
    let neg_eps_simd = f64x8::splat(-hp.eps);

    // signed_dist(u, v) = a*u + b*v + c
    // Use mul_add to encourage FMA and match scalar precision/rounding as closely as possible.
    let dists_vec = a_simd.mul_add(us_simd, b_simd.mul_add(vs_simd, c_simd));
    let mask_simd = dists_vec.simd_ge(neg_eps_simd);
    let full_simd_mask = mask_simd.to_bitmask() as u8;

    // Mask off the bits beyond N.
    // Use u16 for the mask computation to avoid overflow when n == 8.
    let mask = (full_simd_mask as u16 & ((1u16 << {n}) - 1)) as u8;
            "#
        )
    } else {
        format!(
            r#"
    let neg_eps = -hp.eps;

    let mut dists = [0.0f64; {n}];
    let mut mask: u8 = 0;

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();
    unsafe {{
        for i in 0..{n} {{
            let d = hp.signed_dist(*us.add(i), *vs.add(i));
            dists[i] = d;
            mask |= ((d >= neg_eps) as u8) << i;
        }}
    }}
            "#
        )
    };

    let check_and_prepare_mixed = if simd {
        format!(
            r#"
    if mask == 0 {{
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }}

    let full_mask_val: u8 = ((1u16 << {n}) - 1) as u8;
    if mask == full_mask_val {{
        return ClipResult::Unchanged;
    }}
    
    // Mixed case: materialize distances to stack for the jump table accessors.
    let mut dists = [0.0; 8];
    dists_vec.copy_to_slice(&mut dists);
            "#
        )
    } else {
        format!(
            r#"
    if mask == 0 {{
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }}

    let full_mask_val: u8 = ((1u16 << {n}) - 1) as u8;
    if mask == full_mask_val {{
        return ClipResult::Unchanged;
    }}
            "#
        )
    };

    let expanded = format!(
        r#"
#[inline(always)]
fn {fn_name}(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {{
    debug_assert_eq!(poly.len, {n});

    {classification_code}

    {check_and_prepare_mixed}

    out.len = 0;
    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;
    let track_bounding = poly.has_bounding_ref;

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {{
        u.mul_add(u, v * v)
    }}

    #[inline(always)]
    fn get_t(d_in: f64, d_out: f64) -> f64 {{
        d_in / (d_in - d_out)
    }}

    macro_rules! push_v {{
        ($idx:expr) => {{
            let i = $idx;
            let u = poly.us[i];
            let v = poly.vs[i];
            let vp = poly.vertex_planes[i];
            out.push_raw(u, v, vp, poly.edge_planes[i]);
            let r2 = r2_of(u, v);
            if r2 > max_r2 {{
                max_r2 = r2;
            }}
            if track_bounding {{
                has_bounding |= vp.0 == usize::MAX;
            }}
        }};
    }}

    macro_rules! push_entry {{
        ($in_idx:expr, $out_idx:expr, $edge_idx:expr) => {{
            let d_in = dists[$in_idx];
            let d_out = dists[$out_idx];
            let t = get_t(d_in, d_out);
            let u = t.mul_add(poly.us[$out_idx] - poly.us[$in_idx], poly.us[$in_idx]);
            let v = t.mul_add(poly.vs[$out_idx] - poly.vs[$in_idx], poly.vs[$in_idx]);
            let edge_plane = poly.edge_planes[$edge_idx];
            let vp = (edge_plane, hp.plane_idx);
            out.push_raw(u, v, vp, edge_plane);
            let r2 = r2_of(u, v);
            if r2 > max_r2 {{
                max_r2 = r2;
            }}
            if track_bounding {{
                has_bounding |= edge_plane == usize::MAX;
            }}
        }};
    }}

    macro_rules! push_exit {{
        ($in_idx:expr, $out_idx:expr, $edge_idx:expr) => {{
            let d_in = dists[$in_idx];
            let d_out = dists[$out_idx];
            let t = get_t(d_in, d_out);
            let u = t.mul_add(poly.us[$out_idx] - poly.us[$in_idx], poly.us[$in_idx]);
            let v = t.mul_add(poly.vs[$out_idx] - poly.vs[$in_idx], poly.vs[$in_idx]);
            let edge_plane = poly.edge_planes[$edge_idx];
            let vp = (edge_plane, hp.plane_idx);
            out.push_raw(u, v, vp, hp.plane_idx);
            let r2 = r2_of(u, v);
            if r2 > max_r2 {{
                max_r2 = r2;
            }}
            if track_bounding {{
                has_bounding |= edge_plane == usize::MAX;
            }}
        }};
    }}

    #[cold]
    fn fallback(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {{
        clip_convex_small_bool::<{n}>(poly, hp, out)
    }}

    match mask {{
        {match_arms}
        _ => return fallback(poly, hp, out),
    }}

    out.max_r2 = max_r2;
    out.has_bounding_ref = if track_bounding {{ has_bounding }} else {{ false }};
    ClipResult::Changed
}}
"#
    );

    expanded.parse().unwrap()
}

fn parse_ident_and_usize(input: TokenStream) -> Result<(String, usize), TokenStream> {
    let mut it = input.into_iter();
    let name = match it.next() {
        Some(TokenTree::Ident(ident)) => ident.to_string(),
        Some(tt) => {
            return Err(compile_error(&format!(
                "expected function name identifier, got {tt:?}"
            )))
        }
        None => return Err(compile_error("expected `name, N`")),
    };

    match it.next() {
        Some(TokenTree::Punct(p)) if p.as_char() == ',' => {}
        Some(tt) => {
            return Err(compile_error(&format!(
                "expected `,` after function name, got {tt:?}"
            )))
        }
        None => return Err(compile_error("expected `, N` after function name")),
    }

    let n = match it.next() {
        Some(TokenTree::Literal(lit)) => parse_usize_literal(lit)?,
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::None => {
            return Err(compile_error(&format!(
                "unexpected group; expected integer literal, got {g:?}"
            )))
        }
        Some(tt) => {
            return Err(compile_error(&format!(
                "expected integer literal N, got {tt:?}"
            )))
        }
        None => return Err(compile_error("expected integer literal N")),
    };

    // Optional trailing comma.
    if let Some(tt) = it.next() {
        match tt {
            TokenTree::Punct(p) if p.as_char() == ',' => {}
            other => {
                return Err(compile_error(&format!(
                    "unexpected trailing tokens after N: {other:?}"
                )))
            }
        }
    }
    if let Some(other) = it.next() {
        return Err(compile_error(&format!(
            "unexpected trailing tokens after N: {other:?}"
        )));
    }

    Ok((name, n))
}

fn parse_usize_literal(lit: Literal) -> Result<usize, TokenStream> {
    let s = lit.to_string();
    let s = s
        .strip_suffix("usize")
        .or_else(|| s.strip_suffix("u8"))
        .or_else(|| s.strip_suffix("u16"))
        .or_else(|| s.strip_suffix("u32"))
        .or_else(|| s.strip_suffix("u64"))
        .unwrap_or(&s);
    s.parse::<usize>()
        .map_err(|_| compile_error(&format!("could not parse integer literal `{}`", lit)))
}

fn compile_error(msg: &str) -> TokenStream {
    let mut ts = TokenStream::new();
    ts.extend([TokenTree::Ident(Ident::new(
        "compile_error",
        proc_macro::Span::call_site(),
    ))]);
    ts.extend([TokenTree::Punct(Punct::new('!', Spacing::Alone))]);
    let inner = TokenStream::from(TokenTree::Literal(Literal::string(msg)));
    ts.extend([TokenTree::Group(Group::new(Delimiter::Brace, inner))]);
    ts
}
