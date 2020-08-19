from utils.scrape_util import snkrs_img_scraper


snkrs_img_scraper = snkrs_img_scraper()
snkrs_img_scraper.login(username='yrc602@gmail.com', password='y0a6n0g2')
snkrs_img_scraper.search_for_product(keyword='Jordan 1 Retro High Off-White Chicago')
snkrs_img_scraper.get_img_urls_and_names(max_num_scroll=100, num_img=1000)
df = snkrs_img_scraper.img_list_2_df(clear_df=True)
snkrs_img_scraper.download_images(url_list=df.loc[df['title'].str.upper().str.contains('JORDAN|CHICAGO')]['x1'].tolist(),
                                  label_name='Air Jordan 1 Retro High Off-White Chicago')


