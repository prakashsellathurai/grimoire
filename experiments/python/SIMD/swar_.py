a = 0x4567 # this represents four 4-bit unsigned integers, [4, 5, 6, 7]
b = 0x6789 # as does this: [6, 7, 8, 9]

print(hex(a + b))  # 0xacf0 => [0xa, 0xc, 0xf, 0x0] == [10, 12, 15, 0]

# oh no, that's the wrong answer... 6+8 should be 14, not 15.
# it's wrong because the result of 9+7 was 16 (0x10), causing carry propagation
# into the adjacent "lane".

# solution: padding and masking:
a = 0x04050607
b = 0x06070809
m = 0x0f0f0f0f

print(hex((a + b) & m)) # 0xa0c0e00 => [0xa, 0xc, 0xe, 0x0] == [10, 12, 14, 0]