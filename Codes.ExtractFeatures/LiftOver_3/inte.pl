open(A,"CalJac3_summary.txt");
open(B,"CavPor3_summary.txt");
open(C,"EriEur1_summary.txt");
open(D,"GorGor3_summary.txt");
open(E,"Mm10_summary.txt");
open(F,"NomLeu1_summary.txt");
open(G,"OryCun2_summary.txt");
open(H,"PanTro3_summary.txt");
open(I,"PonAbe2_summary.txt");
open(J,"RheMac2_summary.txt");
open(K,"Rn4_summary.txt");
open(L,"SorAra1_summary.txt");
open(M,"TupBel1_summary.txt");
open(O,">position_CRM.txt");
while(<A>){
    @a=split(/\s+/,$_);
    print O "$a[1]\t";
    $l=<B>;
    @a=split(/\s+/,$l);
    print O "$a[1]\t";
    $l=<C>;
    @a=split(/\s+/,$l);
    print O "$a[1]\t";
    $l=<D>;
    @a=split(/\s+/,$l);
    print O "$a[1]\t";
    $l=<E>;
    @a=split(/\s+/,$l);
    print O "$a[1]\t";
    $l=<F>;
    @a=split(/\s+/,$l);
    print O "$a[1]\t";
    $l=<G>;
    @a=split(/\s+/,$l);
    print O "$a[1]\t";
    $l=<H>;
    @a=split(/\s+/,$l);
    print O "$a[1]\t";
    $l=<I>;
    @a=split(/\s+/,$l);
    print O "$a[1]\t";
    $l=<J>;
    @a=split(/\s+/,$l);
    print O "$a[1]\t";
    $l=<K>;
    @a=split(/\s+/,$l);
    print O "$a[1]\t";
    $l=<L>;
    @a=split(/\s+/,$l);
    print O "$a[1]\t";
    $l=<M>;
    @a=split(/\s+/,$l);
    print O "$a[1]\n";
}