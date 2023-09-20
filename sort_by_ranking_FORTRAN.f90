subroutine sort_by_ranking_FORTRAN(A, B, C, N)
  implicit none
  integer :: j
  integer :: N
  real, dimension(N) :: A, B
  real, dimension(N) :: C
  integer :: highest_value_A_idx, highest_value_B_idx
  real :: highest_value_A, highest_value_B

  do j = 1, N
    highest_value_A = maxval(A)   ! find highest value in A
    highest_value_A_idx = maxloc(A, dim=1)   ! find index of highest value in A, USE THIS INDEX
    highest_value_B = maxval(B)   ! find highest value in B, USE THIS VALUE
    highest_value_B_idx = maxloc(B, dim=1)   ! find index of highest value in B

    C(highest_value_A_idx) = highest_value_B   ! replace the highest value in A with the highest value in B

    A(highest_value_A_idx) = -1e6   ! set these to a large negative value to find the next highest value and index
    B(highest_value_B_idx) = -1e6
  end do

end subroutine sort_by_ranking_FORTRAN

