import re

def clean_text(raw):
    t = raw.replace('\r\n', '\n').replace('\r', '\n')

    def is_struct(s):
        return bool(re.match(r'^(\s*([-*+✔✗●▪•→]\s|#{1,6}\s|```|---|\d+[.)]\s*))', s)) \
               or s.strip().startswith('**')

    # Phase 1: merge double-newline word artifacts
    raw_blocks = t.split('\n\n')
    merged = []
    for blk in raw_blocks:
        b = blk.strip()
        if not b:
            continue
        word_count = len(b.split())
        is_artifact = word_count <= 3 and len(b) <= 40 and not is_struct(b)
        if is_artifact and merged:
            prev = merged[-1]
            if not re.search(r'[.!?:;]$', prev.rstrip()):
                merged[-1] = (prev + ' ' + b).replace('  ', ' ').strip()
                continue
        merged.append(b)

    # Phase 2: collapse single-\n within blocks
    final_parts = []
    for block in merged:
        lines = block.split('\n')
        out = []
        buf = []

        def flush():
            if buf:
                out.append(' '.join(buf))
                buf.clear()

        for line in lines:
            trim = line.strip()
            if not trim:
                flush(); out.append('')
            elif is_struct(line):
                flush(); out.append(line)
            else:
                buf.append(trim)
        flush()
        block_result = re.sub(r'\n{3,}', '\n\n', '\n'.join(out)).strip()
        if block_result:
            final_parts.append(block_result)

    return '\n\n'.join(final_parts)


# Test 1: double-newline word-per-para pattern
test1 = (
    "Chung toi nhan thuc rat ro chat luong san pham la dieu kien tien quyet de cong ty ton tai va phat\n\n"
    "trien,\n\nduoc\n\nthe\n\nhien\n\nqua\n\nslogan:\n\n"
    '"A Pioneer of High quality!" (Nha tien phong ve chat luong cao) Gia tri cot loi nay bao trum len...'
)
print("=== TEST 1: double-newline word-per-para ===")
result1 = clean_text(test1)
print(result1)
print()

# Test 2: mix — long sentence cut with \n\n between words at end
test2 = (
    '1.Lang nghe, xin loi, giai thich: "Chung toi xin loi vi su co" giai thich ly do co the xay\n\n'
    "ra\n\ndan\n\nden\n\nviec\n\nhang\n\nhoa\n\nbi\n\nloi/hong\n\nhoc.\n"
    "2.Kiem tra chinh sach cong ty ve van de doi tra hang"
)
print("=== TEST 2: mixed end-of-line + numbered list ===")
print(clean_text(test2))
print()

# Test 3: bullet list should be preserved
test3 = "**file.pdf**\n\n- Item 1\n- Item 2\n- Item 3\n\nSome continuation text"
print("=== TEST 3: bullets preserved ===")
print(clean_text(test3))
print()

# Test 4: real Vietnamese from user screenshot
test4 = (
    "Chung toi nhan thuc rat ro chat luong san pham la dieu kien tien quyet de cong ty ton tai va phat\n\n"
    "trien,\n\nduoc\n\nthe\n\nhien\n\nqua\n\nslogan:\n\n"
    '"A Pioneer of High quality!" (Nha tien phong ve chat luong cao) Gia tri cot loi nay bao trum len cach chung toi suy nghi, cach xay dung y tuong va hanh dong\n\n'
    "dua\n\ntren\n\nnang\n\nluc\n\ncot\n\nloi\n\ncua\n\nchung\n\ntoi\n\nthong\n\nqua\n\ncac\n\nhoat\n\ndong\n\nkinh\n\ndoanh."
)
print("=== TEST 4: full Vietnamese paragraph ===")
print(clean_text(test4))
