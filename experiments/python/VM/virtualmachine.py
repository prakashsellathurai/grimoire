# Virtual Machine from https://third-bit.com/sdxpy/vm/
#   components:
#     -> Instruction pointer - holds memory address of the next instruction to execute It is automatically initisalized to address 0
#     so that every program must start as part of application binary interface  (ABI)
#     -> Four registers from R0 to R3 that can access directly no memory to memory operations
#     -> 256 words of memory each can store single value both program and data live in this single block of memory
#
#   Instruction set defines what it can do they are just numbers ->we write them as assembly code
#  each instruction is 2 byte long  the op_code fits in one byte
# each instrcution is 3 bytes 1 byte for op code 0,1,2 single byte operands
# each operand is a register idebtifier , a constant or an address which is a constant that idntfies a location in memory
# larger number is 256
#
# Table  Virtual machine op codes.
# Name	Code	Format	Action	Example	Equivalent
# hlt	1	--	Halt program	hlt	sys.exit(0)
# ldc	2	rv	Load constant	ldc R0 99	R0 = 99
# ldr	3	rr	Load register	ldr R0 R1	R0 = memory[R1]
# cpy	4	rr	Copy register	cpy R0 R1	R0 = R1
# str	5	rr	Store register	str R0 R1	memory[R1] = R0
# add	6	rr	Add	add R0 R1	R0 = R0 + R1
# sub	7	rr	Subtract	sub R0 R1	R0 = R0 - R1
# xor   8   rr  Xor xor R0 R1 R0 = R0 xor R1
# beq	9	rv	Branch if equal	beq R0 99	if (R0==0) IP = 99
# bne	10	rv	Branch if not equal	bne R0 99	if (R0!=0) IP = 99
# prr	11	r-	Print register	prr R0	print(R0)
# prm	12	r-	Print memory	prm R0	print(memory[R0])
# sta   13  rvv Store Array sta R0 R1 2


NUM_REG = 4  # number of registers
RAM_LEN = 256  # number of words in RAM

OPS = {
    "hlt": {"code": 0x1, "fmt": "--"},  # Halt program
    "ldc": {"code": 0x2, "fmt": "rv"},  # Load value
    "ldr": {"code": 0x3, "fmt": "rr"},  # Load register
    "cpy": {"code": 0x4, "fmt": "rr"},  # Copy register
    "str": {"code": 0x5, "fmt": "rr"},  # Store register
    "add": {"code": 0x6, "fmt": "rr"},  # Add
    "sub": {"code": 0x7, "fmt": "rr"},  # Subtract
    "xor": {"code": 0x8, "fmt": "rr"},  # xor
    "beq": {"code": 0x9, "fmt": "rv"},  # Branch if equal
    "bne": {"code": 0xA, "fmt": "rv"},  # Branch if not equal
    "prr": {"code": 0xB, "fmt": "r-"},  # Print register
    "prm": {"code": 0xC, "fmt": "r-"},  # Print memory
    "sta": {"code": 0xD, "fmt": "rvv"},  # sta R_src base_addr index (Store to array)
    "inc": {"code": 0xE, "fmt": "r-"}, # increment register
    "dec": {"code": 0xF, "fmt": "r-"} #    Decrement Register
}
OP_MASK = 0xFF  # select a single byte
OP_SHIFT = 8  # shift up by one byte
OP_WIDTH = 6  # op width in characters when printing


class VirtualMachine:
    def __init__(self):
        self._initialize([])
        self.prompt = ">>"

    def execute(self, program):
        program = program.split("\n")
        program = [int(op) for op in program]
        self._initialize(program)
        self.run()

    def _initialize(self, program):
        assert len(program) <= RAM_LEN, "Program too long"
        self.ram = [program[i] if (i < len(program)) else 0 for i in range(RAM_LEN)]
        self.ip = 0
        self.reg = [0] * NUM_REG

    def fetch(self):
        instruction = self.ram[self.ip]
        self.ip += 1

        assert self.ip < RAM_LEN, "Program too long"

        op = instruction & OP_MASK
        instruction >>= OP_SHIFT

        arg0 = instruction & OP_MASK
        instruction >>= OP_SHIFT

        arg1 = instruction & OP_MASK
        instruction >>= OP_SHIFT
        
        arg2 = instruction & OP_MASK
        instruction >>= OP_SHIFT

        return [op, arg0, arg1, arg2]

    def run(self):
        running = True

        while running:
            op, arg0, arg1, arg2 = self.fetch()
            if op == OPS["hlt"]["code"]:
                running = False
            elif op == OPS["ldc"]["code"]:
                self.reg[arg0] = arg1
            elif op == OPS["ldr"]["code"]:
                self.reg[arg0] = self.ram[self.reg[arg1]]
            elif op == OPS["cpy"]["code"]:
                self.reg[arg0] = self.reg[arg1]
            elif op == OPS["str"]["code"]:
                self.ram[self.reg[arg1]] = self.reg[arg0]
            elif op == OPS["add"]["code"]:
                self.reg[arg0] += self.reg[arg1]
            elif op == OPS["sub"]["code"]:
                self.reg[arg0] -= self.reg[arg1]
            elif op == OPS["xor"]["code"]:
                self.reg[arg0] ^= self.reg[arg1]
            elif op == OPS["beq"]["code"]:
                if self.reg[arg0] == 0:
                    self.ip = arg1
            elif op == OPS["bne"]["code"]:
                if self.reg[arg0] != 0:
                    self.ip = arg1
            elif op == OPS["prr"]["code"]:
                print(self.reg[arg0])
            elif op == OPS["prm"]["code"]:
                print(self.ram[self.reg[arg0]])
            elif op  == OPS["sta"]["code"]:
                eff_addr = arg1 + arg2
                self.ram[eff_addr] = self.reg[arg0]
            elif op == OPS["inc"]["code"]:
                self.reg[arg0] += 1
            elif op == OPS["dec"]["code"]:
                self.reg[arg0] -= 1

DIVIDER = ".data"


class Assembler:
    def assemble(self, lines):
        lines = self._get_lines(lines)
        to_compile, to_allocate = self._split(lines)

        labels = self._find_labels(lines)
        instructions = [ln for ln in to_compile if not self._is_label(ln)]

        base_of_data = len(instructions)
        self._add_allocations(base_of_data, labels, to_allocate)
        compiled = [self._compile(instr, labels) for instr in instructions]
        program = self._to_text(compiled)
        return program

    def _get_lines(self, lines):
        cleaned = []
        for line in lines.splitlines():
            # remove comments starting with ';'
            line = line.split(";")[0].strip()
            if line:
                 cleaned.append(line)
        return cleaned
    def _find_labels(self, lines):
        result = {}
        loc = 0

        for ln in lines:
            if self._is_label(ln):
                label = ln[:-1].strip()
                assert label not in result, f"Duplicated '{label}'"
                result[label] = loc
            else:
                loc += 1
        return result

    def _is_label(self, line):
        return line.endswith(":")

    def _compile(self, instructions, labels):
        tokens = instructions.split()
        op, args = tokens[0], tokens[1:]
        fmt, code = OPS[op]["fmt"], OPS[op]["code"]

        if fmt == "--":
            return self._combine(code)
        elif fmt == "r-":
            return self._combine(self._reg(args[0]), code)
        elif fmt == "rr":
            return self._combine(self._reg(args[1]), self._reg(args[0]), code)
        elif fmt == "rv":
            return self._combine(self._val(args[1], labels), self._reg(args[0]), code)
        elif fmt == "rvv": # New format for Array Ops
            return self._combine(self._val(args[2], labels), self._val(args[1], labels), self._reg(args[0]), code)
            
    def _combine(self, *args):
        assert len(args) > 0, "cannot combine no arguments"
        result = 0
        for arg in args:
            result <<= OP_SHIFT
            result |= arg
        return result

    def _reg(self, token):
        if token[0] != "R":
            register = int(token)
        else:
            register = int(token[1:])
        assert register < NUM_REG, f"number of register must be less than '{NUM_REG}'"
        return register

    def _val(self, token, labels):
        if token[0] != "@":
            return int(token)
        lbl = token[1:]
        assert lbl in labels, f"Unknown label '{token}'"
        return labels[lbl]

    def _split(self, lines):
        try:
            split = lines.index(DIVIDER)
            return lines[0:split], lines[split + 1 :]
        except ValueError:
            return lines, []

    def _add_allocations(self, base_of_data, labels, to_allocate):
        for alloc in to_allocate:
            fields = [a.strip() for a in alloc.split(":")]
            assert len(fields) == 2, f"invalid allocation directive '{alloc}'"
            lbl, num_words_text = fields
            assert lbl not in labels, f"Duplicate label '{lbl}' in allocation"
            num_words = int(num_words_text)
            assert (base_of_data + num_words) < RAM_LEN, (
                f"Allocation '{lbl}' requires too much memory"
            )
            labels[lbl] = base_of_data
            base_of_data += num_words

    def _to_text(self, compiled):
        return "\n".join(str(op) for op in compiled)


if __name__ == "__main__":
    vm = VirtualMachine()
    print(
        ">> save number 3 to register 0 ; number 4 to register 1; print value at register 1;  print value at register 0 ;halt"
    )
    program = "\n".join(
        str(op) for op in [0x030002, 0x040102, 0x00010B, 0x00000B, 0x000001]
    )
    vm.execute(program)

    print(">> save 100 to register and print it")
    asm = Assembler()
    program = "ldc R1 100\nprr R1\n hlt"
    op_codes = asm.assemble(program)
    vm.execute(op_codes)

    ## # Count up to 3.
    # - R0: loop index.
    # - R1: loop limit.
    print(">> print 0 1 2 using loop")
    code = """ldc R0 0
ldc R1 3
loop:
prr R0
ldc R2 1
add R0 R2
cpy R2 R1
sub R2 R0
bne R2 @loop
hlt"""
    program = asm.assemble(code)
    vm.execute(program)

    print(">> print 0 1 2 using array in loop")
    code = """ldc R0 0
ldc R1 3
ldc R2 @array
loop:
str R0 R2
ldc R3 1
add R0 R3
add R2 R3
cpy R3 R1
sub R3 R0
bne R3 @loop
hlt
.data
array: 10
"""
    program = asm.assemble(code)
    vm.execute(program)
    
    print(">> swap R1 and R2 without affecting other values")
    code = """
ldc R1 0
ldc R2 2
prr R1
prr R2
xor R1 R2
xor R2 R1
xor R1 R2
prr R1
prr R2
hlt
"""
    program = asm.assemble(code)
    vm.execute(program)
    
    print(">> reverse the array")
    code = """
    ldc R2 @array
    
    ldc R0 1
    sta R0 @array 0
    ldc R0 2
    sta R0 @array 1
    ldc R0 3
    sta R0 @array 2
    ldc R0 4
    sta R0 @array 3
    ldc R0 5
    sta R0 @array 4
    
    ldc R0 0
    ldc R1 5
    
    print_before:
    cpy R3 R2
    add R3 R0
    prm R3
    
    ldc R3 1
    add R0 R3
    
    cpy R3 R1
    sub R3 R0
    bne R3 @print_before
    
    ldc R0 0
    ldc R1 4
    
    loop:
    cpy R3 R1
    sub R3 R0
    beq R3 @end
    
    cpy R3 R2
    add R3 R0
    ldr R3 R3
    
    cpy R3 R2
    add R3 R1
    ldr R3 R3
    
    ldc R3 1
    add R0 R3
    sub R1 R3
    
    bne R3 @loop
    
    end:
    
    ldc R0 0
    ldc R1 5
    
    print_after:
    cpy R3 R2
    add R3 R0
    prm R3
    
    ldc R3 1
    add R0 R3
    
    cpy R3 R1
    sub R3 R0
    bne R3 @print_after
    
    hlt
    
    .data
    array: 5
"""
    program = asm.assemble(code)
    vm.execute(program)
    
    print(">> Increment and Decrement")
    code = """ldc R0 0
    ldc R1 3
    prr R1
    inc R1
    prr R1
    dec R1
    prr R1
    hlt"""
    program = asm.assemble(code)
    vm.execute(program)