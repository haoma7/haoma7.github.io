var swiper = new Swiper( '.swiper-container.two', {
    pagination: '.swiper-pagination',
    paginationClickable: true,
    effect: 'coverflow',
    loop: true,
    centeredSlides: true,
    slidesPerView: 'auto',
    coverflow: {
        rotate: 0,
        stretch: 100,
        depth: 150,
        modifier: 1.5,
        slideShadows : false,
    }
} );

const myAtropos = Atropos({
    el: '.my-atropos',
    activeOffset: 40,
    shadowScale: 1.05,
    onEnter() {
      console.log('Enter');
    },
    onLeave() {
      console.log('Leave');
    },
    onRotate(x, y) {
      console.log('Rotate', x, y);
    }
  });